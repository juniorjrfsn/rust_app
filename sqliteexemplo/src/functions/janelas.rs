pub mod janela_mensagem {
    use windows::{
        core::*,
        Win32::UI::WindowsAndMessaging::*
    };
    pub fn open_janela() {
        // println!("Hello, world!");
        unsafe {
            MessageBoxA(None, s!("olá"), s!("World"), MB_OK);
            MessageBoxW(None, w!("como que tá !?"), w!("World"), MB_OK);
        }
    }
}