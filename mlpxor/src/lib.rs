use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuronio {
    pub pesos: Vec<f64>,
    pub bias: f64,
}

impl Neuronio {
    pub fn novo(qtd_entradas: usize) -> Self {
        let mut rng = rand::rng(); // Atualizado: thread_rng() -> rng()
        let limite = (6.0 / (qtd_entradas + 1) as f64).sqrt();
        Neuronio {
            pesos: (0..qtd_entradas).map(|_| rng.random_range(-limite..limite)).collect(), // gen_range -> random_range
            bias: rng.random_range(-limite..limite), // gen_range -> random_range
        }
    }

    pub fn ativar(&self, entradas: &[f64], funcao_ativacao: &str) -> f64 {
        let soma: f64 = entradas.iter().zip(&self.pesos).map(|(x, w)| x * w).sum::<f64>() + self.bias;
        match funcao_ativacao {
            "tanh" => soma.tanh(),
            "sigmoid" => 1.0 / (1.0 + (-soma).exp()),
            _ => panic!("Função não suportada"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedeNeural {
    pub camadas: Vec<Vec<Neuronio>>,
}

impl RedeNeural {

    pub fn inferir_detalhado(&self, entradas: &[f64]) -> f64 {
        println!("\nEntrada: {:?}", entradas);

        let mut ativacoes = entradas.to_vec();

        for (i, camada) in self.camadas.iter().enumerate() {
            let func = if i == self.camadas.len() - 1 {
                "sigmoid"
            } else {
                "tanh"
            };

            let mut novas_ativacoes = Vec::new();

            for (j, neuronio) in camada.iter().enumerate() {
                let valor = neuronio.ativar(&ativacoes, func);

                println!(
                    "  Camada {} - Neurônio {}: {:.6}",
                    i + 1,
                    j + 1,
                    valor
                );

                novas_ativacoes.push(valor);
            }

            ativacoes = novas_ativacoes;
        }

        let saida = ativacoes[0];

        println!("  Saída final: {:.6}", saida);

        saida
    }

    pub fn nova(tamanhos_camadas: &[usize]) -> Self {
        let mut camadas = Vec::new();
        for i in 0..tamanhos_camadas.len() - 1 {
            let num_neuronas = tamanhos_camadas[i + 1];
            let num_entradas = tamanhos_camadas[i];
            let camada = (0..num_neuronas).map(|_| Neuronio::novo(num_entradas)).collect();
            camadas.push(camada);
        }
        RedeNeural { camadas }
    }

    // Passagem direta (Apenas calcula a resposta sem alterar pesos)
    pub fn inferir(&self, entradas: &[f64]) -> f64 {
        let mut ativacoes = entradas.to_vec();
        for (i, camada) in self.camadas.iter().enumerate() {
            let func = if i == self.camadas.len() - 1 { "sigmoid" } else { "tanh" };
            let mut novas_ativacoes = Vec::new();
            for neuronio in camada {
                novas_ativacoes.push(neuronio.ativar(&ativacoes, func));
            }
            ativacoes = novas_ativacoes;
        }
        ativacoes[0]
    }

    fn forward_pass_completo(&self, entradas: &[f64]) -> Vec<Vec<f64>> {
        let mut ativacoes = vec![entradas.to_vec()];
        for (i, camada) in self.camadas.iter().enumerate() {
            let func = if i == self.camadas.len() - 1 { "sigmoid" } else { "tanh" };
            let mut novas_ativacoes = Vec::new();
            for neuronio in camada {
                novas_ativacoes.push(neuronio.ativar(ativacoes.last().unwrap(), func));
            }
            ativacoes.push(novas_ativacoes);
        }
        ativacoes
    }

    // Backpropagation
    pub fn treinar(
    &mut self,
    dados_treino: &[(Vec<f64>, Vec<f64>)],
    taxa_aprendizado: f64,
    epocas: usize,
    ) {
        println!("\n========================================");
        println!("INÍCIO DO TREINAMENTO");
        println!("========================================");
        println!("Taxa de aprendizado: {}", taxa_aprendizado);
        println!("Total de épocas: {}", epocas);
        println!("A primeira época será exibida em detalhes.");
        println!("As demais épocas serão resumidas.\n");

        for epoca in 0..epocas {
            let mut erro_total = 0.0;

            // =========================================================
            // PRIMEIRA ÉPOCA - EXIBIÇÃO DETALHADA
            // =========================================================
            let mostrar_detalhes = epoca == 0;

            if mostrar_detalhes {
                println!("╔════════════════════════════════════════╗");
                println!("║         ÉPOCA 1 - DETALHADA          ║");
                println!("╚════════════════════════════════════════╝");
            }

            for (indice_amostra, (entradas, alvos)) in dados_treino.iter().enumerate() {
                // -----------------------------------------------------
                // 1. FORWARD PASS
                // -----------------------------------------------------
                let ativacoes = self.forward_pass_completo(entradas);
                let saida = ativacoes.last().unwrap();

                let erros_saida: Vec<f64> = saida
                    .iter()
                    .zip(alvos)
                    .map(|(o, t)| t - o)
                    .collect();

                erro_total += erros_saida.iter().map(|e| e * e).sum::<f64>();

                if mostrar_detalhes {
                    println!("\n----------------------------------------");
                    println!("AMOSTRA {}", indice_amostra + 1);
                    println!("----------------------------------------");

                    println!("Entrada : {:?}", entradas);
                    println!("Alvo    : {:?}", alvos);

                    println!("\nAtivações:");

                    for (i, camada) in ativacoes.iter().enumerate() {
                        if i == 0 {
                            println!("  Entrada : {:?}", camada);
                        } else {
                            println!(
                                "  Camada {} : {:?}",
                                i,
                                camada
                                    .iter()
                                    .map(|v| format!("{:.6}", v))
                                    .collect::<Vec<String>>()
                            );
                        }
                    }

                    println!(
                        "\nSaída da rede : {:.6}",
                        saida[0]
                    );

                    println!(
                        "Alvo          : {:.6}",
                        alvos[0]
                    );

                    println!(
                        "Erro (alvo - saída): {:.6}",
                        erros_saida[0]
                    );

                    println!(
                        "Erro²         : {:.6}",
                        erros_saida[0] * erros_saida[0]
                    );
                }

                // -----------------------------------------------------
                // 2. BACKPROPAGATION - DELTA DA SAÍDA
                // -----------------------------------------------------
                let mut deltas = vec![
                    erros_saida
                        .iter()
                        .zip(saida.iter())
                        .map(|(err, out)| err * out * (1.0 - out))
                        .collect::<Vec<f64>>(),
                ];

                if mostrar_detalhes {
                    println!("\nDeltas:");

                    for (j, delta) in deltas[0].iter().enumerate() {
                        println!(
                            "  Saída - Neurônio {}: {:.8}",
                            j + 1,
                            delta
                        );
                    }
                }

                // -----------------------------------------------------
                // 3. BACKPROPAGATION - DELTAS DAS CAMADAS ESCONDIDAS
                // -----------------------------------------------------
                for l in (0..self.camadas.len() - 1).rev() {
                    let mut deltas_camada = Vec::new();

                    for j in 0..self.camadas[l].len() {
                        let mut soma_delta = 0.0;

                        for k in 0..self.camadas[l + 1].len() {
                            soma_delta +=
                                deltas.last().unwrap()[k]
                                    * self.camadas[l + 1][k].pesos[j];
                        }

                        let act = ativacoes[l + 1][j];

                        let delta = soma_delta * (1.0 - act * act);

                        deltas_camada.push(delta);
                    }

                    deltas.push(deltas_camada);
                }

                deltas.reverse();

                if mostrar_detalhes {
                    for (l, camada_deltas) in deltas.iter().enumerate() {
                        println!("  Camada {}:", l + 1);

                        for (j, delta) in camada_deltas.iter().enumerate() {
                            println!(
                                "    Neurônio {}: {:.8}",
                                j + 1,
                                delta
                            );
                        }
                    }
                }

                // -----------------------------------------------------
                // 4. ATUALIZAÇÃO DOS PESOS E BIAS
                // -----------------------------------------------------
                if mostrar_detalhes {
                    println!("\nAtualização dos pesos:");
                }

                for l in 0..self.camadas.len() {
                    for j in 0..self.camadas[l].len() {
                        for k in 0..self.camadas[l][j].pesos.len() {
                            let peso_anterior = self.camadas[l][j].pesos[k];

                            let ajuste =
                                taxa_aprendizado
                                    * deltas[l][j]
                                    * ativacoes[l][k];

                            self.camadas[l][j].pesos[k] += ajuste;

                            if mostrar_detalhes {
                                println!(
                                    "  Camada {} | Neurônio {} | Peso {}: \
                                    {:.8} -> {:.8}  (Δ = {:+.8})",
                                    l + 1,
                                    j + 1,
                                    k + 1,
                                    peso_anterior,
                                    self.camadas[l][j].pesos[k],
                                    ajuste
                                );
                            }
                        }

                        let bias_anterior = self.camadas[l][j].bias;

                        let ajuste_bias =
                            taxa_aprendizado * deltas[l][j];

                        self.camadas[l][j].bias += ajuste_bias;

                        if mostrar_detalhes {
                            println!(
                                "  Camada {} | Neurônio {} | Bias: \
                                {:.8} -> {:.8}  (Δ = {:+.8})",
                                l + 1,
                                j + 1,
                                bias_anterior,
                                self.camadas[l][j].bias,
                                ajuste_bias
                            );
                        }
                    }
                }

                if mostrar_detalhes {
                    println!("\n[OK] Atualização da amostra concluída.");
                }
            }

            // =========================================================
            // RESUMO DA ÉPOCA
            // =========================================================
            let erro_medio = erro_total / dados_treino.len() as f64;

            if mostrar_detalhes {
                println!("\n========================================");
                println!("FIM DA ÉPOCA 1");
                println!("Erro médio: {:.8}", erro_medio);
                println!("========================================\n");
            } else if (epoca + 1) % 10000 == 0 {
                println!(
                    "Época {:>5} | Erro médio = {:.8}",
                    epoca + 1,
                    erro_medio
                );
            }
        }

        println!("\n========================================");
        println!("TREINAMENTO FINALIZADO");
        println!("========================================\n");
    }

    pub fn salvar_modelo(&self, caminho: &str) {
        let json = serde_json::to_string_pretty(self).unwrap();
        let mut arquivo = File::create(caminho).unwrap();
        arquivo.write_all(json.as_bytes()).unwrap();
        println!("\n[OK] Modelo treinado e salvo em '{}'!", caminho);
    }

    pub fn carregar_modelo(caminho: &str) -> Self {
        let mut arquivo = File::open(caminho).expect("\n[ERRO] Modelo não encontrado! Treine a rede primeiro.");
        let mut json = String::new();
        arquivo.read_to_string(&mut json).unwrap();
        serde_json::from_str(&json).unwrap()
    }
}