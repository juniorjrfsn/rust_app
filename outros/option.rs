// An integer division that doesn't `panic!`
fn checked_division(dividend: i32, divisor: i32) -> Option<i32> {
    if divisor == 0 {
        None  // Failure is represented as the `None` variant
    } else {
        Some(dividend / divisor)  // Result is wrapped in a `Some` variant
    }
}

// This function handles a division that may not succeed
fn try_division(dividend: i32, divisor: i32) {
    // `Option` values can be pattern matched, just like other enums
    match checked_division(dividend, divisor) {
        None => println!("{} / {} failed!", dividend, divisor),
        Some(quotient) => {
            println!("{} / {} = {}", dividend, divisor, quotient)
        },
    }
}
fn get_user_name(user_id: u32) -> Option<String> {
    // Simulate fetching user data from a database
    let users = [
      (1, String::from("Alice")),
      (2, String::from("Bob")),
      (3, String::from("Charlie")),
    ];
  
    // Find the user matching the ID
    for (id, name) in users.iter() {
      if *id == user_id {
        return Some(name.clone()); // Return Some(String) if found
      }
    }
  
    // User not found, return None
    None
  }
fn main() {
    let user_name = get_user_name(4);

    // Handle the potential absence of a username
    match user_name {
      Some(name) => println!("User name: {}", name),
      None => println!("User not found!"),
    }

    try_division(4, 2);
    try_division(1, 3);

    // Binding `None` to a variable needs to be type annotated
    let none: Option<i32> = None;
    let _equivalent_none = None::<i32>;

    let optional_float = Some(0f32);

    // Unwrapping a `Some` variant will extract the value wrapped.
    println!("{:?} unwraps to {:?}", optional_float, optional_float.unwrap());

    // Unwrapping a `None` variant will `panic!`
    println!("{:?} unwraps to {:?}", none, none.unwrap());
}

//  cd outros 
/*
.\option.bat
rustc .\option.rs
 .\option.exe

*/ 