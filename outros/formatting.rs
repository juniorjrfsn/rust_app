use std::fmt::{self, Formatter, Display};

#[derive(Debug)]
struct Arvore {
    especie: String,
    tamanho: f32,
    idade: u8,
}
impl Display for Arvore { 
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let tamanho_c   = if self.tamanho > 0.0 { 'm'       } else { 'c'        };
        let idade_c     = if self.idade >= 20   { "Antiga"  } else { "Jovem"    };
        write!(f, "{}: altura {:.3}{}, idade : {:.3}anos, {}", self.especie, self.tamanho.abs(), tamanho_c, self.idade, idade_c)
    }
}

#[derive(Debug)]
struct City {
    name: &'static str,
    // Latitude
    lat: f32,
    // Longitude
    lon: f32,
}

impl Display for City {
    // `f` is a buffer, and this method must write the formatted string into it.
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let lat_c = if self.lat >= 0.0 { 'N' } else { 'S' };
        let lon_c = if self.lon >= 0.0 { 'E' } else { 'W' };

        // `write!` is like `format!`, but it will write the formatted string
        // into a buffer (the first argument).
        write!(f, "{}: {:.3}?{} {:.3}?{}",
               self.name, self.lat.abs(), lat_c, self.lon.abs(), lon_c)
    }
}

#[derive(Debug)]
struct Color {
    red: u8,
    green: u8,
    blue: u8,
}

fn main() {
    for arvore in [
        Arvore { especie: "Manga".to_string(),      tamanho: 53.347778, idade: 5 },
        Arvore { especie: "Abacate".to_string(),    tamanho: 59.95,     idade: 40 },
        Arvore { especie: "Ipe".to_string(),        tamanho: 49.25,     idade: 10 },
    ] {
        println!("{}", arvore);
    }

    for city in [
        City { name: "Dublin", lat: 53.347778, lon: -6.259722 },
        City { name: "Oslo", lat: 59.95, lon: 10.75 },
        City { name: "Vancouver", lat: 49.25, lon: -123.1 },
    ] {
        println!("{}", city);
    }
    for color in [
        Color { red: 128, green: 255, blue: 90 },
        Color { red: 0, green: 3, blue: 254 },
        Color { red: 0, green: 0, blue: 0 },
    ] {
        // Switch this to use {} once you've added an implementation
        // for fmt::Display.
        println!("{:?}", color);
    }
}

// rustc .\formatting.rs
// ./formatting.exe