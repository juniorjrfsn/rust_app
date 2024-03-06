pub mod ger {
	use rand::Rng;

	pub fn gera_aposta()  -> String {
		let mut rng = rand::thread_rng(); 
		// Crie uma string vazia para armazenar a cadeia de caracteres.
		let mut string = String::new(); 
		// Gere 6 dígitos aleatórios.
 
		let mut cont = 0;
		for _ in 0..6 {
			for _ in 0..2 {
				let digit = rng.gen_range(0..10);
				string.push_str(&digit.to_string());
				// Imprima a string gerada.
				// println!("{}", string);
			}
			cont = cont+1;
			if cont < 6 {
				string.push_str(&"-")
			}
		} 
		string
	}
}