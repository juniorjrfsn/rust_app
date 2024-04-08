
fn adole(){
    println!("A teen,(midle)")
}
fn maior(){
    println!("A high,(adults)")
}
fn imortal(){
    println!("A imortal,(advanced)")
}
fn main() {
    let number = 100;
    // TODO ^ Try different values for `number`

    println!("Tell me about {}", number);
    match number {
        // Match a single value
        1 => println!("One!"),
        // Match several values
        2 | 3 | 5 | 7 | 11 => println!("This is a prime"),
        // TODO ^ Try adding 13 to the list of prime values
        // Match an inclusive range
        13..=19 => adole() ,
        20..=119 => maior() ,
        120..=1000 => imortal() ,
        // Handle the rest of cases
        _ => println!("Ain't special"),
        // TODO ^ Try commenting out this catch-all arm
    }

    let boolean = true;
    // Match is an expression too
    let binary = match boolean {
        // The arms of a match must cover all the possible values
        false => 0,
        true => 1,
        // TODO ^ Try commenting out one of these arms
    };

    println!("{} -> {}", boolean, binary);
    println!("Valor máximo de f32: {}", f32::MAX);
}

//  cd outros 
/*
.\match.bat
rustc .\match.rs
 .\match.exe

*/ 