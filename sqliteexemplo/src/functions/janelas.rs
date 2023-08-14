pub mod janelaMensagem {
    use windows::{
        core::*,
        Win32::UI::WindowsAndMessaging::*
    };
    pub fn openJanela() {
        // println!("Hello, world!");
        unsafe {
            MessageBoxA(None, s!("olá"), s!("World"), MB_OK);
            MessageBoxW(None, w!("como que tá !?"), w!("World"), MB_OK);
        }
    }
}