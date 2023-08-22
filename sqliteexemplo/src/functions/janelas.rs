pub mod janela_mensagem {
    use windows::{
        core::*,
        Win32::UI::WindowsAndMessaging::*
    };
    pub fn open_janela() {
        // println!("Hello, world!");
        unsafe {
            MessageBoxA(None, s!("olá manada"), s!("World do mundo"), MB_OK);
            MessageBoxW(None, w!("como que tá !? World del mundo"), w!("World del mundo"), MB_OK);
        }
    }
}