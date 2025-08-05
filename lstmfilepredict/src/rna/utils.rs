
// projeto: lstmfilepredict
// file: src/rna/utils.rs


pub mod utils {
    pub fn mask_password(conn_str: &str) -> String {
        if let Some(at_pos) = conn_str.find('@') {
            if let Some(colon_pos) = conn_str[..at_pos].find(':') {
                let start = conn_str[..colon_pos].to_string();
                let end = conn_str[at_pos..].to_string();
                format!("{}:***{}", start, end)
            } else {
                conn_str.to_string()
            }
        } else {
            conn_str.to_string()
        }
    }
}