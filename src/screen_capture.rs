use std::{
    ffi::{CString, c_void},
    fmt::Debug,
    mem::{size_of, zeroed},
    ptr::{self, NonNull, null_mut},
    sync::OnceLock,
    thread,
    time::Duration,
};

use half::f16;
use libloading::Library;
use ndarray::Ix4;
use ort::{
    memory::{AllocationDevice, AllocatorType, MemoryType},
    sys::OrtMemoryInfo,
    tensor::{IntoTensorElementType, TensorElementType},
    value::{Tensor, Value},
};
use windows::{
    Win32::{
        Foundation::HMODULE,
        Graphics::{
            Direct3D::{
                D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_11_1,
                Fxc::{D3DCOMPILE_OPTIMIZATION_LEVEL3, D3DCompile},
            },
            Direct3D11::{
                D3D11_BIND_CONSTANT_BUFFER, D3D11_BIND_SHADER_RESOURCE, D3D11_BIND_UNORDERED_ACCESS, D3D11_BOX, D3D11_BUFFER_DESC, D3D11_BUFFER_UAV, D3D11_CPU_ACCESS_FLAG, D3D11_CPU_ACCESS_READ, D3D11_CPU_ACCESS_WRITE, D3D11_CREATE_DEVICE_FLAG, D3D11_MAP_READ, D3D11_RESOURCE_MISC_FLAG,
                D3D11_SDK_VERSION, D3D11_SUBRESOURCE_DATA, D3D11_TEXTURE2D_DESC, D3D11_UAV_DIMENSION_BUFFER, D3D11_UNORDERED_ACCESS_VIEW_DESC, D3D11_UNORDERED_ACCESS_VIEW_DESC_0, D3D11_USAGE_DEFAULT, D3D11_USAGE_DYNAMIC, D3D11_USAGE_STAGING, D3D11CreateDevice, ID3D11Buffer, ID3D11ComputeShader,
                ID3D11Device, ID3D11DeviceContext, ID3D11Resource, ID3D11Texture2D, ID3D11UnorderedAccessView,
            },
            Dxgi::{
                Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R32_UINT},
                CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, DXGI_ERROR_ACCESS_DENIED, DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_WAIT_TIMEOUT, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, DXGI_OUTDUPL_FRAME_INFO, IDXGIAdapter4, IDXGIDevice4, IDXGIFactory6, IDXGIOutput6, IDXGIOutputDuplication,
            },
        },
        System::Com::{COINIT_MULTITHREADED, CoInitializeEx},
        UI::HiDpi::{DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext},
    },
    core::{Interface, PCSTR},
};

use crate::errors::ScreenCaptureError;

#[repr(C)]
#[derive(Copy, Clone)]
struct CaptureRegion {
    capture_size: (u32, u32),
    _unused: [u32; 2],
}

pub enum ScreenCaptureOutput<'a> {
    Available(ScreenCaptureOutputAvailable<'a>),
    NotAvailable(ScreenCaptureOutputAvailable<'a>),
}

pub struct MappedTensor<T: IntoTensorElementType + Debug> {
    pub tensor: Option<Tensor<T>>,
    _cuda_resource: CudaGraphicsResource,
}

impl<T: IntoTensorElementType + Debug> MappedTensor<T> {
    pub fn new_mapped(tensor: Tensor<T>, cuda_resource: CudaGraphicsResource) -> Self {
        Self { tensor: Some(tensor), _cuda_resource: cuda_resource }
    }

    pub fn take_tensor(&mut self) -> Option<Tensor<T>> {
        std::mem::take(&mut self.tensor)
    }
}

impl<T: IntoTensorElementType + Debug> Drop for MappedTensor<T> {
    fn drop(&mut self) {
        cuda_graphics_unmap_resources(1, &self._cuda_resource as *const CudaGraphicsResource as *mut CudaGraphicsResource, 0);
    }
}

pub struct ScreenCaptureOutputAvailable<'a> {
    underlying_context: ID3D11DeviceContext,
    staging_buffer: ID3D11Buffer,
    preallocated_buffer: &'a mut Vec<f16>,
    data: ID3D11Buffer,
    cuda_resource: CudaGraphicsResource,
    width: u32,
    height: u32,
}

impl<'a> ScreenCaptureOutputAvailable<'a> {
    pub fn data_as_ndarray(&mut self) -> ndarray::ArrayView<f16, Ix4> {
        unsafe { self.underlying_context.CopyResource(&self.staging_buffer, &self.data) };
        let mapped_resource = unsafe {
            let mut mapped_resource = zeroed();
            self.underlying_context.Map(&self.staging_buffer, 0, D3D11_MAP_READ, 0, Some(&mut mapped_resource)).expect("Failed to map resource");
            mapped_resource
        };
        let p_data = mapped_resource.pData as *const f16;
        let slice_of_pie = unsafe { std::slice::from_raw_parts(p_data, 3 * self.width as usize * self.height as usize) };
        self.preallocated_buffer.copy_from_slice(slice_of_pie);
        let output = unsafe { ndarray::ArrayView::from_shape_ptr((1, 3, self.width as usize, self.height as usize), self.preallocated_buffer.as_mut_ptr()) };
        unsafe { self.underlying_context.Unmap(&self.staging_buffer, 0) };
        output
    }

    pub unsafe fn data_as_ort_tensor(&self) -> MappedTensor<f16> {
        cuda_graphics_map_resources(1, &self.cuda_resource as *const CudaGraphicsResource as *mut CudaGraphicsResource, 0);

        let mut p_data = ptr::null_mut();
        let p_size = ptr::null_mut();
        let st = cuda_graphics_resource_get_mapped_pointer(&mut p_data as *mut *mut _, p_size, self.cuda_resource);
        assert_eq!(st, CUDA_SUCCESS, "Failed to get mapped pointer: {}", st);

        let mut p_ort_value = ptr::null_mut();
        let shape = [1, 3, self.height as i64, self.width as i64];
        let mut memory_info_ptr: *mut OrtMemoryInfo = std::ptr::null_mut();
        let allocator_name = CString::new(AllocationDevice::CUDA_PINNED.as_str()).unwrap_or_else(|_| unreachable!());
        let m = ort::api().CreateMemoryInfo;
        let st = unsafe { m(allocator_name.as_ptr(), AllocatorType::Arena.into(), 0, MemoryType::Default.into(), &mut memory_info_ptr) };
        assert_eq!(st.0, null_mut());
        let m = ort::api().CreateTensorWithDataAsOrtValue;
        let st = unsafe {
            m(
                memory_info_ptr as *const _,
                p_data,
                (self.width * self.height * 3 * std::mem::size_of::<f16>() as u32) as usize,
                shape.as_ptr(),
                shape.len(),
                TensorElementType::Float16.into(),
                &mut p_ort_value,
            )
        };
        assert_eq!(st.0, null_mut());

        MappedTensor::new_mapped(unsafe { Value::from_ptr(NonNull::new_unchecked(p_ort_value), None) }, self.cuda_resource)
    }
}

pub struct ScreenCapturer {
    _adapter: IDXGIAdapter4,
    _output: IDXGIOutput6,
    duplication: Option<IDXGIOutputDuplication>,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    _capture_region_buffer: ID3D11Buffer,
    captured_frame_texture: ID3D11Texture2D,
    output_buffer: ID3D11Buffer,
    output_buffer_cuda_resource: CudaGraphicsResource,
    _compute_shader: ID3D11ComputeShader,
    _uav: ID3D11UnorderedAccessView,
    staging_buffer: ID3D11Buffer,
    pub screen_width: u32,
    pub screen_height: u32,
    capture_width: u32,
    capture_height: u32,
    center_box: D3D11_BOX,
    preallocated_buffer: Vec<f16>,
}

impl ScreenCapturer {
    pub fn create(gpu_index: u32, monitor_index: u32, capture_width: u32, capture_height: u32) -> Result<Self, ScreenCaptureError> {
        unsafe {
            CoInitializeEx(None, COINIT_MULTITHREADED).ok().map_err(|_| ScreenCaptureError::RuntimeInitializationError)?;
            let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

            let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
            let adapter = factory.EnumAdapterByGpuPreference::<IDXGIAdapter4>(gpu_index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE).map_err(|_| ScreenCaptureError::ApiInitializationError)?;
            let output: IDXGIOutput6 = adapter.EnumOutputs(monitor_index).map_err(|_| ScreenCaptureError::ApiInitializationError)?.cast().unwrap();

            let (screen_width, screen_height) = {
                let desc = output.GetDesc().unwrap();
                ((desc.DesktopCoordinates.right - desc.DesktopCoordinates.left) as u32, (desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top) as u32)
            };

            let center_box = D3D11_BOX {
                left: (screen_width - capture_width) as u32 / 2,
                top: (screen_height - capture_height) as u32 / 2,
                front: 0,
                right: (screen_width + capture_width) as u32 / 2,
                bottom: (screen_height + capture_height) as u32 / 2,
                back: 1,
            };

            let mut feature_level: D3D_FEATURE_LEVEL = Default::default();
            let mut device = None;
            let mut ctx = None;

            D3D11CreateDevice(
                &adapter,
                D3D_DRIVER_TYPE_UNKNOWN,
                HMODULE(null_mut()),
                D3D11_CREATE_DEVICE_FLAG(0),
                Some(&[D3D_FEATURE_LEVEL_11_1]),
                D3D11_SDK_VERSION,
                Some(&mut device),
                Some(&mut feature_level),
                Some(&mut ctx),
            )
            .map_err(|_| ScreenCaptureError::ApiInitializationError)?;

            let device = device.unwrap();
            let context = ctx.unwrap();

            let dxgi_device = device.cast::<IDXGIDevice4>().unwrap();
            dxgi_device.SetGPUThreadPriority(7).map_err(|_| ScreenCaptureError::ApiInitializationError)?;

            let duplication = Some(output.DuplicateOutput(&device).map_err(|_| ScreenCaptureError::ApiInitializationError)?);
            let capture_region_buffer = create_capture_region_buffer(&device, capture_width, capture_height)?;
            let output_buffer = create_output_buffer(&device, capture_width, capture_height)?;
            let output_buffer_cuda_resource = register_cuda_resource(&output_buffer)?;
            let captured_frame_texture = create_captured_frame_texture(&device, capture_width, capture_height)?;
            let compute_shader = create_compute_shader(&device)?;
            let uav = create_uav(&device, &output_buffer, capture_width, capture_height)?;
            let staging_buffer = create_staging_buffer(&device, capture_width, capture_height)?;

            context.CSSetConstantBuffers(0, Some(&[Some(capture_region_buffer.clone())]));
            context.CSSetUnorderedAccessViews(0, 1, Some(&Some(uav.clone())), None);
            context.CSSetShader(Some(&compute_shader), None);

            Ok(Self {
                _adapter: adapter,
                _output: output,
                duplication,
                device,
                context,
                _capture_region_buffer: capture_region_buffer,
                captured_frame_texture,
                output_buffer,
                output_buffer_cuda_resource,
                _compute_shader: compute_shader,
                _uav: uav,
                staging_buffer,
                screen_width,
                screen_height,
                capture_width,
                capture_height,
                center_box,
                preallocated_buffer: vec![f16::ZERO; 3 * capture_width as usize * capture_height as usize],
            })
        }
    }

    pub fn capture_frame(&mut self) -> Result<ScreenCaptureOutput, ScreenCaptureError> {
        unsafe {
            if self.duplication.is_none() {
                return Err(ScreenCaptureError::DuplicationNotAvailable);
            }

            let dupl = self.duplication.as_ref().unwrap();

            let frame_resource = {
                let mut frame_info = DXGI_OUTDUPL_FRAME_INFO::default();
                let mut frame_resource = None;
                match dupl.AcquireNextFrame(0, &mut frame_info, &mut frame_resource) {
                    Ok(_) => {}
                    Err(e) => {
                        if e.code() == DXGI_ERROR_ACCESS_LOST || e.code() == DXGI_ERROR_ACCESS_DENIED {
                            self.reacquire_duplication()?;
                            return Err(ScreenCaptureError::DXGIAccessLost(e));
                        } else if e.code() == DXGI_ERROR_WAIT_TIMEOUT {
                            return Ok(ScreenCaptureOutput::NotAvailable(ScreenCaptureOutputAvailable {
                                underlying_context: self.context.clone(),
                                staging_buffer: self.staging_buffer.clone(),
                                data: self.output_buffer.clone(),
                                cuda_resource: self.output_buffer_cuda_resource,
                                preallocated_buffer: &mut self.preallocated_buffer,
                                width: self.capture_width,
                                height: self.capture_height,
                            }));
                        }
                    }
                }
                frame_resource.unwrap()
            };

            let frame_d3d11_resource = frame_resource.cast::<ID3D11Resource>().unwrap();
            self.context.CopySubresourceRegion(&self.captured_frame_texture, 0, 0, 0, 0, Some(&frame_d3d11_resource), 0, Some(&self.center_box));

            let shader_resource = {
                let mut shader_resource = None;
                self.device.CreateShaderResourceView(&self.captured_frame_texture, None, Some(&mut shader_resource)).map_err(|e| ScreenCaptureError::ShaderResorceCreationError(e))?;
                shader_resource.ok_or(ScreenCaptureError::ShaderResorceCreationError(windows::core::Error::from_win32()))?
            };
            self.context.CSSetShaderResources(0, Some(&[Some(shader_resource)]));

            let dispatch_x = (self.capture_width as f32 / 8.0).ceil() as u32;
            let dispatch_y = (self.capture_height as f32 / 8.0).ceil() as u32;
            self.context.Dispatch(dispatch_x, dispatch_y, 1);

            dupl.ReleaseFrame().unwrap();

            Ok(ScreenCaptureOutput::Available(ScreenCaptureOutputAvailable {
                underlying_context: self.context.clone(),
                staging_buffer: self.staging_buffer.clone(),
                data: self.output_buffer.clone(),
                cuda_resource: self.output_buffer_cuda_resource,
                preallocated_buffer: &mut self.preallocated_buffer,
                width: self.capture_width,
                height: self.capture_height,
            }))
        }
    }

    pub fn reacquire_duplication(&mut self) -> Result<(), ScreenCaptureError> {
        unsafe {
            if self.duplication.is_some() {
                self.duplication = None;
            }
            while self.duplication.is_none() {
                thread::sleep(Duration::from_millis(1000));
                let output = self._output.DuplicateOutput(&self.device);
                self.duplication = output.ok();
            }

            Ok(())
        }
    }
}

unsafe fn create_capture_region_buffer(device: &ID3D11Device, capture_width: u32, capture_height: u32) -> Result<ID3D11Buffer, ScreenCaptureError> {
    unsafe {
        let mut capture_region_buffer = None;
        device
            .CreateBuffer(
                &D3D11_BUFFER_DESC {
                    ByteWidth: size_of::<CaptureRegion>() as u32,
                    Usage: D3D11_USAGE_DYNAMIC,
                    BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                    CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
                    MiscFlags: 0,
                    StructureByteStride: 0,
                },
                Some(&D3D11_SUBRESOURCE_DATA {
                    pSysMem: &CaptureRegion {
                        capture_size: (capture_width, capture_height),
                        _unused: [0, 0],
                    } as *const CaptureRegion as *const _,
                    ..Default::default()
                }),
                Some(&mut capture_region_buffer),
            )
            .map_err(|_| ScreenCaptureError::CaptureRegionBufferCreationError)?;

        Ok(capture_region_buffer.ok_or(ScreenCaptureError::CaptureRegionBufferCreationError)?)
    }
}

unsafe fn create_output_buffer(device: &ID3D11Device, capture_width: u32, capture_height: u32) -> Result<ID3D11Buffer, ScreenCaptureError> {
    unsafe {
        let mut output_buffer = None;
        device
            .CreateBuffer(
                &D3D11_BUFFER_DESC {
                    ByteWidth: capture_width * capture_height * 3 * size_of::<f16>() as u32,
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: D3D11_BIND_UNORDERED_ACCESS.0 as u32 | D3D11_BIND_SHADER_RESOURCE.0 as u32,
                    CPUAccessFlags: 0,
                    MiscFlags: 0,
                    StructureByteStride: 0,
                },
                None,
                Some(&mut output_buffer),
            )
            .map_err(|_| ScreenCaptureError::OutputBufferCreationError)?;

        Ok(output_buffer.ok_or(ScreenCaptureError::OutputBufferCreationError)?)
    }
}

unsafe fn create_captured_frame_texture(device: &ID3D11Device, capture_width: u32, capture_height: u32) -> Result<ID3D11Texture2D, ScreenCaptureError> {
    unsafe {
        let mut texture = None;
        device
            .CreateTexture2D(
                &D3D11_TEXTURE2D_DESC {
                    Width: capture_width,
                    Height: capture_height,
                    MipLevels: 1,
                    ArraySize: 1,
                    Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                    CPUAccessFlags: D3D11_CPU_ACCESS_FLAG(0).0 as u32,
                    MiscFlags: D3D11_RESOURCE_MISC_FLAG(0).0 as u32,
                },
                None,
                Some(&mut texture),
            )
            .map_err(|_| ScreenCaptureError::InputTextureCreationError)?;

        Ok(texture.ok_or(ScreenCaptureError::InputTextureCreationError)?)
    }
}

fn create_compute_shader(device: &ID3D11Device) -> Result<ID3D11ComputeShader, ScreenCaptureError> {
    let mut compiled_shader_code = None;
    let source_name = CString::new("BgraTextureToRgbTensor").unwrap();
    let entry_point = CString::new("CSMain").unwrap();
    let target = CString::new("cs_5_0").unwrap();

    unsafe {
        D3DCompile(
            SHADER_CODE.as_ptr() as *const _,
            SHADER_CODE.len(),
            PCSTR(source_name.as_ptr() as *const _),
            None,
            None,
            PCSTR(entry_point.as_ptr() as *const _),
            PCSTR(target.as_ptr() as *const _),
            D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut compiled_shader_code,
            None,
        )
        .map_err(|_| ScreenCaptureError::ShaderCompilationError)
    }?;

    let compiled_shader_code = compiled_shader_code.unwrap();

    let compiled_shader_code = unsafe { std::slice::from_raw_parts(compiled_shader_code.GetBufferPointer() as *const u8, compiled_shader_code.GetBufferSize() as usize) };

    let mut compute_shader = None;
    unsafe { device.CreateComputeShader(compiled_shader_code, None, Some(&mut compute_shader)).map_err(|_| ScreenCaptureError::ShaderCreationError)? };

    Ok(compute_shader.ok_or(ScreenCaptureError::ShaderCreationError)?)
}

fn create_uav(device: &ID3D11Device, buffer: &ID3D11Buffer, capture_width: u32, capture_height: u32) -> Result<ID3D11UnorderedAccessView, ScreenCaptureError> {
    unsafe {
        let mut uav = None;
        device
            .CreateUnorderedAccessView(
                buffer,
                Some(&D3D11_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_UINT,
                    ViewDimension: D3D11_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D11_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D11_BUFFER_UAV {
                            FirstElement: 0,
                            NumElements: 3 * capture_width * capture_height / 2, // we're packing accessing 2 pixels at a time for packing.
                            Flags: 0,
                        },
                    },
                }),
                Some(&mut uav),
            )
            .map_err(|_| ScreenCaptureError::UAVCreationError)?;

        Ok(uav.ok_or(ScreenCaptureError::UAVCreationError)?)
    }
}

fn create_staging_buffer(device: &ID3D11Device, capture_width: u32, capture_height: u32) -> Result<ID3D11Buffer, ScreenCaptureError> {
    unsafe {
        let mut staging_buffer = None;
        device
            .CreateBuffer(
                &D3D11_BUFFER_DESC {
                    ByteWidth: capture_width * capture_height * 3 * size_of::<f16>() as u32,
                    Usage: D3D11_USAGE_STAGING,
                    BindFlags: 0,
                    CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32 | D3D11_CPU_ACCESS_READ.0 as u32,
                    MiscFlags: 0,
                    StructureByteStride: 0,
                },
                None,
                Some(&mut staging_buffer),
            )
            .map_err(|_| ScreenCaptureError::OutputBufferCreationError)?;

        Ok(staging_buffer.ok_or(ScreenCaptureError::OutputBufferCreationError)?)
    }
}

fn register_cuda_resource(buffer: &ID3D11Buffer) -> Result<CudaGraphicsResource, ScreenCaptureError> {
    let mut cuda_resource = None;
    let result = cuda_graphics_d3d11_register_resource(&mut cuda_resource, buffer.cast::<ID3D11Resource>().unwrap(), CUDA_RESOURCE_REGISTER_FLAG_NONE);
    match result {
        CUDA_SUCCESS => Ok(cuda_resource.unwrap()),
        _ => Err(ScreenCaptureError::CudaRegistrationError(result)),
    }
}

const SHADER_CODE: &'static str = r#"
    cbuffer CaptureRegion : register(b0)
    {
        int2 captureSize;
    };

    RWStructuredBuffer<uint> output : register(u0);
    Texture2D<float4> inputTexture : register(t0);

    [numthreads(8, 8, 1)]
    void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
    {
        if (dispatchThreadID.x >= captureSize.x || dispatchThreadID.y >= captureSize.y)
        {
            return;
        }

        int2 texCoord = dispatchThreadID.xy;
        float4 color = inputTexture.Load(int3(texCoord, 0));

        int width = captureSize.x;
        int height = captureSize.y;

        int linearIndex = dispatchThreadID.y * width + dispatchThreadID.x;

        int bufferIndexR = linearIndex / 2;
        int bufferIndexG = (height * width + linearIndex) / 2;
        int bufferIndexB = (2 * height * width + linearIndex) / 2;


        output[bufferIndexR] = 0;
        output[bufferIndexG] = 0;
        output[bufferIndexB] = 0;

        uint packedR, packedG, packedB;
        if (linearIndex % 2 == 0)
        {
            packedR = asuint(f32tof16(color.r)) & 0xFFFF;
            packedG = asuint(f32tof16(color.g)) & 0xFFFF;
            packedB = asuint(f32tof16(color.b)) & 0xFFFF;
        }
        else
        {
            packedR = (asuint(f32tof16(color.r)) & 0xFFFF) << 16;
            packedG = (asuint(f32tof16(color.g)) & 0xFFFF) << 16;
            packedB = (asuint(f32tof16(color.b)) & 0xFFFF) << 16;
        }

        InterlockedOr(output[bufferIndexR], packedR);
        InterlockedOr(output[bufferIndexG], packedG);
        InterlockedOr(output[bufferIndexB], packedB);
    }
"#;

unsafe impl Send for ScreenCapturer {}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CudaGraphicsResource(pub NonNull<()>);

type CudaGraphicsD3d11RegisterResourceFn = unsafe extern "C" fn(*mut Option<CudaGraphicsResource>, ID3D11Resource, u32) -> i32;
type CudaGraphicsMapResourcesFn = unsafe extern "C" fn(u32, *mut CudaGraphicsResource, u32) -> i32;
type CudaGraphicsUnmapResourcesFn = unsafe extern "C" fn(u32, *mut CudaGraphicsResource, u32) -> i32;
type CudaGraphicsResourceGetMappedPointerFn = unsafe extern "C" fn(*mut *mut c_void, *mut usize, CudaGraphicsResource) -> i32;

pub static LIB_NVCUDA: OnceLock<Library> = OnceLock::new();

pub fn get_lib_nvcuda() -> &'static Library {
    LIB_NVCUDA.get_or_init(|| unsafe { Library::new("cudart64_12.dll").unwrap() })
}

fn cuda_graphics_d3d11_register_resource(cu_resource: *mut Option<CudaGraphicsResource>, d3d_resource: ID3D11Resource, flags: u32) -> i32 {
    let register_cuda_resource: libloading::Symbol<CudaGraphicsD3d11RegisterResourceFn> = unsafe { get_lib_nvcuda().get(b"cudaGraphicsD3D11RegisterResource").unwrap() };
    unsafe { register_cuda_resource(cu_resource, d3d_resource, flags) }
}

fn cuda_graphics_map_resources(count: u32, resources: *mut CudaGraphicsResource, stream: u32) -> i32 {
    let map_resources: libloading::Symbol<CudaGraphicsMapResourcesFn> = unsafe { get_lib_nvcuda().get(b"cudaGraphicsMapResources").unwrap() };
    unsafe { map_resources(count, resources, stream) }
}

fn cuda_graphics_unmap_resources(count: u32, resources: *mut CudaGraphicsResource, stream: u32) -> i32 {
    let unmap_resources: libloading::Symbol<CudaGraphicsUnmapResourcesFn> = unsafe { get_lib_nvcuda().get(b"cudaGraphicsUnmapResources").unwrap() };
    unsafe { unmap_resources(count, resources, stream) }
}

fn cuda_graphics_resource_get_mapped_pointer(p_data: *mut *mut c_void, p_size: *mut usize, resource: CudaGraphicsResource) -> i32 {
    let get_mapped_pointer: libloading::Symbol<CudaGraphicsResourceGetMappedPointerFn> = unsafe { get_lib_nvcuda().get(b"cudaGraphicsResourceGetMappedPointer").unwrap() };
    unsafe { get_mapped_pointer(p_data, p_size, resource) }
}

pub const CUDA_RESOURCE_REGISTER_FLAG_NONE: u32 = 0;
pub const CUDA_RESOURCE_REGISTER_FLAG_TEXTURE_GATHER: u32 = 0x02;
pub const CUDA_RESOURCE_REGISTER_FLAG_SURFACE_LOAD_STORE: u32 = 0x08;

pub const CUDA_SUCCESS: i32 = 0;
pub const CUDA_ERROR_INVALID_DEVICE: i32 = 101;
pub const CUDA_ERROR_INVALID_VALUE: i32 = 11;
pub const CUDA_ERROR_INVALID_RESOURCE_HANDLE: i32 = 400;
pub const CUDA_ERROR_NOT_UNKOWN: i32 = 999;
