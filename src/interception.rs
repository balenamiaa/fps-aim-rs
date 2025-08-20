use std::ffi::c_ulong;

use serde::{Deserialize, Serialize};

use crate::errors::AimAssistError;

#[allow(non_camel_case_types)]
pub type InterceptionContextRaw = *mut std::ffi::c_void;

#[allow(non_camel_case_types)]
pub type InterceptionDevice = i32;

#[allow(non_camel_case_types)]
pub type InterceptionPrecedence = i32;

#[allow(non_camel_case_types)]
pub type InterceptionFilter = u16;

#[allow(non_camel_case_types)]
pub type InterceptionPredicate = Option<unsafe extern "C" fn(device: InterceptionDevice) -> i32>;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct InterceptionMouseStroke {
    pub state: u16,
    pub flags: u16,
    pub rolling: i16,
    pub x: i32,
    pub y: i32,
    pub information: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct InterceptionKeyStroke {
    pub code: u16,
    pub state: u16,
    pub information: u32,
}

#[allow(non_camel_case_types)]
pub type InterceptionStroke = [u8; std::mem::size_of::<InterceptionMouseStroke>()];

pub const INTERCEPTION_SCANCODE_SPACE: u16 = 0x39;

pub const INTERCEPTION_MAX_KEYBOARD: i32 = 10;
pub const INTERCEPTION_MAX_MOUSE: i32 = 10;
pub const INTERCEPTION_MAX_DEVICE: i32 = INTERCEPTION_MAX_KEYBOARD + INTERCEPTION_MAX_MOUSE;

pub const INTERCEPTION_KEY_DOWN: u16 = 0x00;
pub const INTERCEPTION_KEY_UP: u16 = 0x01;
pub const INTERCEPTION_KEY_E0: u16 = 0x02;
pub const INTERCEPTION_KEY_E1: u16 = 0x04;
pub const INTERCEPTION_KEY_TERMSRV_SET_LED: u16 = 0x08;
pub const INTERCEPTION_KEY_TERMSRV_SHADOW: u16 = 0x10;
pub const INTERCEPTION_KEY_TERMSRV_VKPACKET: u16 = 0x20;

pub const INTERCEPTION_FILTER_KEY_NONE: u16 = 0x0000;
pub const INTERCEPTION_FILTER_KEY_ALL: u16 = 0xFFFF;
pub const INTERCEPTION_FILTER_KEY_DOWN: u16 = INTERCEPTION_KEY_UP;
pub const INTERCEPTION_FILTER_KEY_UP: u16 = INTERCEPTION_KEY_UP << 1;
pub const INTERCEPTION_FILTER_KEY_E0: u16 = INTERCEPTION_KEY_E0 << 1;
pub const INTERCEPTION_FILTER_KEY_E1: u16 = INTERCEPTION_KEY_E1 << 1;
pub const INTERCEPTION_FILTER_KEY_TERMSRV_SET_LED: u16 = INTERCEPTION_KEY_TERMSRV_SET_LED << 1;
pub const INTERCEPTION_FILTER_KEY_TERMSRV_SHADOW: u16 = INTERCEPTION_KEY_TERMSRV_SHADOW << 1;
pub const INTERCEPTION_FILTER_KEY_TERMSRV_VKPACKET: u16 = INTERCEPTION_KEY_TERMSRV_VKPACKET << 1;

pub const INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN: u16 = 0x001;
pub const INTERCEPTION_MOUSE_LEFT_BUTTON_UP: u16 = 0x002;
pub const INTERCEPTION_MOUSE_RIGHT_BUTTON_DOWN: u16 = 0x004;
pub const INTERCEPTION_MOUSE_RIGHT_BUTTON_UP: u16 = 0x008;
pub const INTERCEPTION_MOUSE_MIDDLE_BUTTON_DOWN: u16 = 0x010;
pub const INTERCEPTION_MOUSE_MIDDLE_BUTTON_UP: u16 = 0x020;

pub const INTERCEPTION_MOUSE_BUTTON_1_DOWN: u16 = INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN;
pub const INTERCEPTION_MOUSE_BUTTON_1_UP: u16 = INTERCEPTION_MOUSE_LEFT_BUTTON_UP;
pub const INTERCEPTION_MOUSE_BUTTON_2_DOWN: u16 = INTERCEPTION_MOUSE_RIGHT_BUTTON_DOWN;
pub const INTERCEPTION_MOUSE_BUTTON_2_UP: u16 = INTERCEPTION_MOUSE_RIGHT_BUTTON_UP;
pub const INTERCEPTION_MOUSE_BUTTON_3_DOWN: u16 = INTERCEPTION_MOUSE_MIDDLE_BUTTON_DOWN;
pub const INTERCEPTION_MOUSE_BUTTON_3_UP: u16 = INTERCEPTION_MOUSE_MIDDLE_BUTTON_UP;

pub const INTERCEPTION_MOUSE_BUTTON_4_DOWN: u16 = 0x040;
pub const INTERCEPTION_MOUSE_BUTTON_4_UP: u16 = 0x080;
pub const INTERCEPTION_MOUSE_BUTTON_5_DOWN: u16 = 0x100;
pub const INTERCEPTION_MOUSE_BUTTON_5_UP: u16 = 0x200;

pub const INTERCEPTION_MOUSE_WHEEL: u16 = 0x400;
pub const INTERCEPTION_MOUSE_HWHEEL: u16 = 0x800;

pub const INTERCEPTION_FILTER_MOUSE_NONE: u16 = 0x0000;
pub const INTERCEPTION_FILTER_MOUSE_ALL: u16 = 0xFFFF;

pub const INTERCEPTION_FILTER_MOUSE_LEFT_BUTTON_DOWN: u16 = INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN;
pub const INTERCEPTION_FILTER_MOUSE_LEFT_BUTTON_UP: u16 = INTERCEPTION_MOUSE_LEFT_BUTTON_UP;
pub const INTERCEPTION_FILTER_MOUSE_RIGHT_BUTTON_DOWN: u16 = INTERCEPTION_MOUSE_RIGHT_BUTTON_DOWN;
pub const INTERCEPTION_FILTER_MOUSE_RIGHT_BUTTON_UP: u16 = INTERCEPTION_MOUSE_RIGHT_BUTTON_UP;
pub const INTERCEPTION_FILTER_MOUSE_MIDDLE_BUTTON_DOWN: u16 = INTERCEPTION_MOUSE_MIDDLE_BUTTON_DOWN;
pub const INTERCEPTION_FILTER_MOUSE_MIDDLE_BUTTON_UP: u16 = INTERCEPTION_MOUSE_MIDDLE_BUTTON_UP;

pub const INTERCEPTION_FILTER_MOUSE_BUTTON_1_DOWN: u16 = INTERCEPTION_MOUSE_BUTTON_1_DOWN;
pub const INTERCEPTION_FILTER_MOUSE_BUTTON_1_UP: u16 = INTERCEPTION_MOUSE_BUTTON_1_UP;
pub const INTERCEPTION_FILTER_MOUSE_BUTTON_2_DOWN: u16 = INTERCEPTION_MOUSE_BUTTON_2_DOWN;
pub const INTERCEPTION_FILTER_MOUSE_BUTTON_2_UP: u16 = INTERCEPTION_MOUSE_BUTTON_2_UP;
pub const INTERCEPTION_FILTER_MOUSE_BUTTON_3_DOWN: u16 = INTERCEPTION_MOUSE_BUTTON_3_DOWN;
pub const INTERCEPTION_FILTER_MOUSE_BUTTON_3_UP: u16 = INTERCEPTION_MOUSE_BUTTON_3_UP;

pub const INTERCEPTION_FILTER_MOUSE_BUTTON_4_DOWN: u16 = INTERCEPTION_MOUSE_BUTTON_4_DOWN;
pub const INTERCEPTION_FILTER_MOUSE_BUTTON_4_UP: u16 = INTERCEPTION_MOUSE_BUTTON_4_UP;
pub const INTERCEPTION_FILTER_MOUSE_BUTTON_5_DOWN: u16 = INTERCEPTION_MOUSE_BUTTON_5_DOWN;
pub const INTERCEPTION_FILTER_MOUSE_BUTTON_5_UP: u16 = INTERCEPTION_MOUSE_BUTTON_5_UP;

pub const INTERCEPTION_FILTER_MOUSE_WHEEL: u16 = INTERCEPTION_MOUSE_WHEEL;
pub const INTERCEPTION_FILTER_MOUSE_HWHEEL: u16 = INTERCEPTION_MOUSE_HWHEEL;

pub const INTERCEPTION_FILTER_MOUSE_MOVE: u16 = 0x1000;

pub const INTERCEPTION_MOUSE_MOVE_RELATIVE: u16 = 0x000;
pub const INTERCEPTION_MOUSE_MOVE_ABSOLUTE: u16 = 0x001;
pub const INTERCEPTION_MOUSE_VIRTUAL_DESKTOP: u16 = 0x002;
pub const INTERCEPTION_MOUSE_ATTRIBUTES_CHANGED: u16 = 0x004;
pub const INTERCEPTION_MOUSE_MOVE_NOCOALESCE: u16 = 0x008;
pub const INTERCEPTION_MOUSE_TERMSRV_SRC_SHADOW: u16 = 0x100;

// type InterceptionCreateContextFn = unsafe extern "C" fn() -> InterceptionContextRaw;
// type InterceptionDestroyContextFn = unsafe extern "C" fn(context: InterceptionContextRaw);
// type InterceptionGetPrecedenceFn = unsafe extern "C" fn(context: InterceptionContextRaw, device: InterceptionDevice) -> InterceptionPrecedence;
// type InterceptionSetPrecedenceFn = unsafe extern "C" fn(context: InterceptionContextRaw, device: InterceptionDevice, precedence: InterceptionPrecedence);
// type InterceptionGetFilterFn = unsafe extern "C" fn(context: InterceptionContextRaw, device: InterceptionDevice) -> InterceptionFilter;
// type InterceptionSetFilterFn = unsafe extern "C" fn(context: InterceptionContextRaw, predicate: InterceptionPredicate, filter: InterceptionFilter);
// type InterceptionWaitFn = unsafe extern "C" fn(context: InterceptionContextRaw) -> InterceptionDevice;
// type InterceptionWaitWithTimeoutFn = unsafe extern "C" fn(context: InterceptionContextRaw, milliseconds: u32) -> InterceptionDevice;
// type InterceptionSendFn = unsafe extern "C" fn(context: InterceptionContextRaw, device: InterceptionDevice, stroke: *const InterceptionStroke, nstroke: u32) -> i32;
// type InterceptionReceiveFn = unsafe extern "C" fn(context: InterceptionContextRaw, device: InterceptionDevice, stroke: *mut InterceptionStroke, nstroke: u32) -> i32;
// type InterceptionGetHardwareIdFn = unsafe extern "C" fn(context: InterceptionContextRaw, device: InterceptionDevice, hardware_id_buffer: *mut c_void, buffer_size: u32) -> u32;
// type InterceptionIsInvalidFn = unsafe extern "C" fn(device: InterceptionDevice) -> i32;
// type InterceptionIsKeyboardFn = unsafe extern "C" fn(device: InterceptionDevice) -> i32;
// type InterceptionIsMouseFn = unsafe extern "C" fn(device: InterceptionDevice) -> i32;

// pub static LIB_INTERCEPTION: OnceLock<Library> = OnceLock::new();

// pub fn get_lib_interception() -> &'static Library {
//     LIB_INTERCEPTION.get_or_init(|| unsafe { Library::new("interception.dll").unwrap() })
// }

// pub unsafe fn interception_create_context() -> InterceptionContextRaw {
//     let func: Symbol<InterceptionCreateContextFn> = get_lib_interception().get(b"interception_create_context\0").unwrap();
//     func()
// }

// pub unsafe fn interception_destroy_context(context: InterceptionContextRaw) {
//     let func: Symbol<InterceptionDestroyContextFn> = get_lib_interception().get(b"interception_destroy_context\0").unwrap();
//     func(context);
// }

// pub unsafe fn interception_get_precedence(context: InterceptionContextRaw, device: InterceptionDevice) -> InterceptionPrecedence {
//     let func: Symbol<InterceptionGetPrecedenceFn> = get_lib_interception().get(b"interception_get_precedence\0").unwrap();
//     func(context, device)
// }

// pub unsafe fn interception_set_precedence(context: InterceptionContextRaw, device: InterceptionDevice, precedence: InterceptionPrecedence) {
//     let func: Symbol<InterceptionSetPrecedenceFn> = get_lib_interception().get(b"interception_set_precedence\0").unwrap();
//     func(context, device, precedence);
// }

// pub unsafe fn interception_get_filter(context: InterceptionContextRaw, device: InterceptionDevice) -> InterceptionFilter {
//     let func: Symbol<InterceptionGetFilterFn> = get_lib_interception().get(b"interception_get_filter\0").unwrap();
//     func(context, device)
// }

// pub unsafe fn interception_set_filter(context: InterceptionContextRaw, predicate: InterceptionPredicate, filter: InterceptionFilter) {
//     let func: Symbol<InterceptionSetFilterFn> = get_lib_interception().get(b"interception_set_filter\0").unwrap();
//     func(context, predicate, filter);
// }

// pub unsafe fn interception_wait(context: InterceptionContextRaw) -> InterceptionDevice {
//     let func: Symbol<InterceptionWaitFn> = get_lib_interception().get(b"interception_wait\0").unwrap();
//     func(context)
// }

// pub unsafe fn interception_wait_with_timeout(context: InterceptionContextRaw, milliseconds: c_ulong) -> InterceptionDevice {
//     let func: Symbol<InterceptionWaitWithTimeoutFn> = get_lib_interception().get(b"interception_wait_with_timeout\0").unwrap();
//     func(context, milliseconds as u32)
// }

// pub unsafe fn interception_send(context: InterceptionContextRaw, device: InterceptionDevice, stroke: *const InterceptionStroke, nstroke: u32) -> i32 {
//     let func: Symbol<InterceptionSendFn> = get_lib_interception().get(b"interception_send\0").unwrap();
//     func(context, device, stroke, nstroke)
// }

// pub unsafe fn interception_receive(context: InterceptionContextRaw, device: InterceptionDevice, stroke: *mut InterceptionStroke, nstroke: u32) -> i32 {
//     let func: Symbol<InterceptionReceiveFn> = get_lib_interception().get(b"interception_receive\0").unwrap();
//     func(context, device, stroke, nstroke)
// }

// pub unsafe fn interception_get_hardware_id(context: InterceptionContextRaw, device: InterceptionDevice, hardware_id_buffer: *mut std::ffi::c_void, buffer_size: u32) -> u32 {
//     let func: Symbol<InterceptionGetHardwareIdFn> = get_lib_interception().get(b"interception_get_hardware_id\0").unwrap();
//     func(context, device, hardware_id_buffer, buffer_size)
// }

// pub unsafe fn interception_is_invalid(device: InterceptionDevice) -> i32 {
//     let func: Symbol<InterceptionIsInvalidFn> = get_lib_interception().get(b"interception_is_invalid\0").unwrap();
//     func(device)
// }

// pub unsafe fn interception_is_keyboard(device: InterceptionDevice) -> i32 {
//     let func: Symbol<InterceptionIsKeyboardFn> = get_lib_interception().get(b"interception_is_keyboard\0").unwrap();
//     func(device)
// }

// pub unsafe fn interception_is_mouse(device: InterceptionDevice) -> i32 {
//     let func: Symbol<InterceptionIsMouseFn> = get_lib_interception().get(b"interception_is_mouse\0").unwrap();
//     func(device)
// }

#[link(name = "interception", kind = "dylib")]
unsafe extern "C" {
    pub(crate) fn interception_create_context() -> InterceptionContextRaw;
    pub(crate) fn interception_destroy_context(context: InterceptionContextRaw);
    pub(crate) fn interception_get_precedence(context: InterceptionContextRaw, device: InterceptionDevice) -> InterceptionPrecedence;
    pub(crate) fn interception_set_precedence(context: InterceptionContextRaw, device: InterceptionDevice, precedence: InterceptionPrecedence);
    pub(crate) fn interception_get_filter(context: InterceptionContextRaw, device: InterceptionDevice) -> InterceptionFilter;
    pub(crate) fn interception_set_filter(context: InterceptionContextRaw, predicate: InterceptionPredicate, filter: InterceptionFilter);
    pub(crate) fn interception_wait(context: InterceptionContextRaw) -> InterceptionDevice;
    pub(crate) fn interception_wait_with_timeout(context: InterceptionContextRaw, milliseconds: c_ulong) -> InterceptionDevice;
    pub(crate) fn interception_send(context: InterceptionContextRaw, device: InterceptionDevice, stroke: *const InterceptionStroke, nstroke: u32) -> i32;
    pub(crate) fn interception_receive(context: InterceptionContextRaw, device: InterceptionDevice, stroke: *mut InterceptionStroke, nstroke: u32) -> i32;
    pub(crate) fn interception_get_hardware_id(context: InterceptionContextRaw, device: InterceptionDevice, hardware_id_buffer: *mut std::ffi::c_void, buffer_size: u32) -> u32;
    pub(crate) fn interception_is_invalid(device: InterceptionDevice) -> i32;
    pub(crate) fn interception_is_keyboard(device: InterceptionDevice) -> i32;
    pub(crate) fn interception_is_mouse(device: InterceptionDevice) -> i32;
}

pub struct InterceptionContext {
    context: *mut std::ffi::c_void,
    mouse_device: i32,
    keyboard_device: i32,
}

impl InterceptionContext {
    pub fn new() -> Self {
        let context = unsafe { interception_create_context() };
        let mouse_device = Self::get_mouse_device();
        let keyboard_device = Self::get_keyboard_device();
        Self { context, mouse_device, keyboard_device }
    }

    fn get_mouse_device() -> i32 {
        let mut device = 1;
        loop {
            if device >= INTERCEPTION_MAX_DEVICE {
                panic!("No mouse device found");
            }

            if unsafe { interception_is_mouse(device) } != 0 {
                log::info!("Mouse device found: {}", device);
                return device;
            }
            device += 1;
        }
    }

    fn get_keyboard_device() -> i32 {
        let mut device = 0;
        loop {
            if device >= INTERCEPTION_MAX_DEVICE {
                panic!("No keyboard device found");
            }

            if unsafe { interception_is_keyboard(device) } != 0 {
                log::info!("Keyboard device found: {}", device);
                return device;
            }
            device += 1;
        }
    }

    pub fn move_relative(&self, dx: i32, dy: i32) -> Result<(), AimAssistError> {
        let stroke = InterceptionMouseStroke {
            state: 0,
            flags: INTERCEPTION_MOUSE_MOVE_RELATIVE,
            rolling: 0,
            x: dx,
            y: dy,
            information: 0,
        };

        let result = unsafe { interception_send(self.context, self.mouse_device, &stroke as *const _ as *const InterceptionStroke, 1) };

        if result == 1 { Ok(()) } else { Err(AimAssistError::MouseMovementError) }
    }

    pub fn send_key(&self, key: u16, state: u16) -> Result<(), AimAssistError> {
        let stroke = InterceptionKeyStroke { code: key, state, information: 0 };

        let result = unsafe { interception_send(self.context, self.keyboard_device, &stroke as *const _ as *const InterceptionStroke, 1) };

        if result == 1 { Ok(()) } else { Err(AimAssistError::MouseMovementError) }
    }
}

impl Drop for InterceptionContext {
    fn drop(&mut self) {
        unsafe { interception_destroy_context(self.context) };
    }
}

pub fn is_button_held_mouse5() -> bool {
    let state = unsafe { GetAsyncKeyState(0x06) };
    state != 0
}

pub fn is_button_held_mouse4() -> bool {
    let state = unsafe { GetAsyncKeyState(0x05) };
    state != 0
}

pub fn is_vk_down(vk: i32) -> bool {
    let state = unsafe { GetAsyncKeyState(vk) };
    state != 0
}

#[link(name = "user32")]
unsafe extern "system" {
    fn GetAsyncKeyState(vkey: i32) -> i16;
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum VirtualKey {
    LButton = 0x01,
    RButton = 0x02,
    MButton = 0x04,
    X1Button = 0x05,
    X2Button = 0x06,
}
