pub mod codes {

    use std::collections::HashMap;

    use std::ops::{Index, IndexMut};
    pub struct Vector3d<T> {
        pub x: T,
        pub y: T,
        pub z: T,
    }
    impl<T> Index<usize> for Vector3d<T> {
        type Output = T;
        fn index(&self, index: usize) -> &T {
            match index {
                0 => &self.x,
                1 => &self.y,
                2 => &self.z,
                n => panic!("Invalid Vector3d index: {}", n),
            }
        }
    }
    impl<T> IndexMut<usize> for Vector3d<T> {
        fn index_mut(&mut self, index: usize) -> &mut T {
            match index {
                0 => &mut self.x,
                1 => &mut self.y,
                2 => &mut self.z,
                n => panic!("Invalid Vector3d index: {}", n),
            }
        }
    }

    fn get_cds_strs(_x: bool)  {

        let mut state_codes: HashMap<&str, &str> = HashMap::new();
        state_codes.insert("NV", "Nevada");
        state_codes.insert("NY", "New York");

        for (key, val) in state_codes.iter() {
            println!("key: {key} val: {val}");
        }
        println!("................................");

        println!("........ filtro de palavras que tenha a letra a .......");
        let palavras: Vec<&str> = vec![
            "casa", "carro", "árvore", "cidade", "cão", "gato", "flor", "mar", "lua", "sol",
        ];
        let palavras_com_a: Vec<&str> = palavras
            .iter()
            .filter(|&&palavra| palavra.contains('a'))
            .cloned()
            .collect();
        println!("{:?}", palavras_com_a);

        println!("................................");
        let numbers: [String; 3] = [
            "1 DHFGUTY".to_string(),
            "2TYUTY".to_string(),
            "3YT456".to_string(),
        ];
        numbers.iter().for_each(|x: &String| println!("{}", x));

        println!("................................");
        let names: [&str; 3] = ["Sam", "Janet", "Hunter"];
        let csv = names.join(" - ");
        println!("{}", csv);
        /* let vect: Vec<_> = names.into_iter().map(ToOwned::to_owned).collect();*/
        println!("................................");
        let obj = HashMap::from([("key1", "value1"), ("key2", "value2")]);
        for prop in obj.keys() {
            println!("{}: {}", prop, obj.get(prop).unwrap());
        }
    }

    pub fn get_codes_string(printar:Option<bool>) -> () {
       return match printar {
            Some(p) => {
                match p {
                            true => get_cds_strs(p),
                            //_ => println!("Geracao de Strings negado"),
                            _ => println!(""),
                        }
            },
            None => {
                println!("Valor nao informado")
            },
        }
    }
    fn get_cds_(_x: bool)  {
        println!("................................");
        let map = HashMap::from([("a", 1), ("b", 2), ("c", 3)]);
        for (key, val) in map.iter() {
            println!("key: {key} val: {val}");
        }

        println!("................................");
        let numbers = [1, 2, 3, 4, 5];
        for number in numbers {
            println!("{}", number);
        }

        println!(".............. find first ..................");
        let numbers: [i32; 5] = [1, 2, 4, 5, 6];
        let first_even: Option<&i32> = numbers.iter().find(|x: &&i32| *x % 3 == 0);
        println!("{:?}", first_even.unwrap());

        println!("................................");
        let my_array1: [i32; 5] = [1, 2, 3, 4, 5];
        let mut index: usize = 0;
        while index < my_array1.len() {
            println!("{}", my_array1[index]);
            index += 1;
        }
        println!("................................");

        let my_array2 = [1, 2, 3, 4, 5];
        for item in my_array2.iter() {
            println!("{}", item);
        }
        println!("................................");

        let my_array: [i32; 5] = [1, 2, 3, 4, 5];
        for (item, index) in my_array.iter().enumerate() {
            println!("{}: {}", index, item);
        }
        println!("................................");

        let my_vec: Vec<u8> = vec![72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]; // "Hello World" in ASCII
        let vec_to_string = String::from_utf8(my_vec).unwrap(); // Converting to string
        println!("{}", vec_to_string); // Output: Hello World
    }


    pub fn get_codes(printar:Option<bool>) -> ()   {
        return match printar {
             Some(p) => match p { true => get_cds_(p),  _ => println!("Geracao de numeros negado"),} ,
             None =>  println!("Valor nao informado") ,
         }
     }

}
