pub mod janela_mensagem {
 
    use std::io::Error;
    use windows::{
        core::*,
        Win32::UI::WindowsAndMessaging::*
    };

    use std::ffi::CString;
    // use std::ptr;
    use std::ptr::null_mut as NULL;

    pub fn open_janela() {
        // println!("Hello, world!");
        let lp_text = CString::new("Hello, world!").unwrap();
        let hello: String = String::from("Hello, world!") ;
        // let hello: Vec<u16> = "Wassa wassa wassup\0".encode_utf16().collect();
        let lp_caption = CString::new("MessageBox Example").unwrap();

        let hello = String::from("Hello, world! agora siim").to_owned();
        //let vec:Vec<u16> = hello.bytes();

        let ret = unsafe {
            let wide: Vec<u16> = Vec::new();
            let bytes: [u8; 7] = [1, 2, 3, 4, 5, 6, 7];
            let (prefix, shorts, suffix) = bytes.align_to::<u16>();


            let hello = String::from("Hello, world! agora siim");
            let bytes = hello.as_bytes();
            let utf16_units: Vec<u16> = bytes.iter().map(|byte| *byte as u16).collect();
            
            println!("{:?}", utf16_units);
            println!("prefix:{:?} shorts:{:?}  suffix:{:?} ", prefix, shorts, suffix  );
        };

        unsafe {
            // MessageBoxA(ptr::null_mut(),  hello.as_ptr(), lp_caption, MB_OK);
            MessageBoxA(None, s!("olah galera"), s!("World do mundo"), MB_OK);
            MessageBoxW(None, w!("como que tá !? World del mundo"), w!("World del mundo"), MB_OK);
            /*
                MessageBoxA(
                    None,
                    lp_text.as_ptr(),
                    lp_caption.as_ptr(),
                    MB_OK | MB_ICONINFORMATION
                );
            */
        }
    }
}