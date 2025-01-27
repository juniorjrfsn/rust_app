use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configurações do gráfico
    let root = BitMapBackend::new("mlp_graph.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let (input_layer, hidden_layer, output_layer) = (4, 5, 3); // Número de neurônios em cada camada

    // Desenhar os neurônios da camada de entrada
    for i in 0..input_layer {
        root.draw(&Circle::new(
            (100, 100 + i * 100),
            20,
            Into::<ShapeStyle>::into(&BLUE).filled(),
        ))?;
        root.draw(&Text::new(
            format!("Input {}", i + 1),
            (70, 100 + i * 100),
            ("sans-serif", 15).into_font().color(&BLACK),
        ))?;
    }

    // Desenhar os neurônios da camada oculta
    for i in 0..hidden_layer {
        root.draw(&Circle::new(
            (300, 100 + i * 80),
            20,
            Into::<ShapeStyle>::into(&GREEN).filled(),
        ))?;
        root.draw(&Text::new(
            format!("Hidden {}", i + 1),
            (270, 100 + i * 80),
            ("sans-serif", 15).into_font().color(&BLACK),
        ))?;
    }

    // Desenhar os neurônios da camada de saída
    for i in 0..output_layer {
        root.draw(&Circle::new(
            (500, 150 + i * 100),
            20,
            Into::<ShapeStyle>::into(&RED).filled(),
        ))?;
        root.draw(&Text::new(
            format!("Output {}", i + 1),
            (530, 150 + i * 100),
            ("sans-serif", 15).into_font().color(&BLACK),
        ))?;
    }

    // Conectar os neurônios da camada de entrada à camada oculta
    for i in 0..input_layer {
        for j in 0..hidden_layer {
            root.draw(&PathElement::new(
                vec![
                    (120, 100 + i * 100),
                    (280, 100 + j * 80),
                ],
                Into::<ShapeStyle>::into(&BLACK).stroke_width(1),
            ))?;
        }
    }

    // Conectar os neurônios da camada oculta à camada de saída
    for i in 0..hidden_layer {
        for j in 0..output_layer {
            root.draw(&PathElement::new(
                vec![
                    (320, 100 + i * 80),
                    (480, 150 + j * 100),
                ],
                Into::<ShapeStyle>::into(&BLACK).stroke_width(1),
            ))?;
        }
    }

    // Adicionar título
    root.draw(&Text::new(
        "Multilayer Perceptron (MLP)",
        (200, 20),
        ("sans-serif", 20).into_font().color(&BLACK),
    ))?;

    Ok(())
}


// cd mlp
// cargo run
// cargo run --bin mlp
// cargo.exe "run", "--package", "mlp", "--bin", "mlp"