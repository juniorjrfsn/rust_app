WITH Entrada AS (
    SELECT column1 AS ptext, column2 AS ordem
    FROM (VALUES ('Quanto',1), ('que',2), ('é',3), ('1',4), ('+',5), ('1',6), ('?',7)  ) AS ptexts
),
EntradaOrdenada AS (
    SELECT DISTINCT p2.rid, p2.fid, p2.pid, P1.ordem, P1.ptext
        , ROW_NUMBER()  OVER (PARTITION  BY  p2.rid, p2.fid ORDER BY  p2.rid ASC, p2.fid ASC, P1.ordem ASC ) AS entradaordem
        , COUNT(p2.fid)  OVER (PARTITION  BY p2.fid, p2.rid ORDER BY p2.fid ASC, p2.rid ASC) AS tamanho_fid
        , COUNT(P1.ordem)  OVER (PARTITION  BY p2.fid, p2.rid ORDER BY p2.fid ASC, p2.rid ASC) AS tamanho_entrada
    FROM Entrada P1
    INNER JOIN pergunta p2 ON( P1.ptext = p2.ptext)
    GROUP BY p2.rid, p2.fid, p2.pid, P1.ordem, P1.ptext ORDER BY p2.rid ASC, p2.fid ASC, P1.ordem ASC,  p2.pid ASC
),
PerguntaOrdenada AS (
    SELECT DISTINCT p2.rid, p2.fid, p2.pid,  p2.ptext
    ,  ROW_NUMBER()  OVER (PARTITION  BY  p2.rid, p2.fid ORDER BY  p2.rid ASC, p2.fid ASC) AS perguntaordem
    FROM EntradaOrdenada eo
    INNER JOIN pergunta p2 ON(eo.rid = p2.rid AND eo.fid = p2.fid AND eo.ptext = p2.ptext) ORDER BY p2.rid ASC
),
Filtro AS (
    SELECT  po.*
    FROM PerguntaOrdenada po
    INNER JOIN EntradaOrdenada eo ON(po.rid = eo.rid AND po.fid = eo.fid AND po.ptext = eo.ptext AND po.perguntaordem = eo.entradaordem)
    ORDER BY po.rid ASC
),
Relevanciaresposta AS (
    SELECT DISTINCT re.rid, re.fid, re.ptext, COUNT(re.rid) OVER(PARTITION BY re.rid, re.fid ) AS relevancia_frase
    FROM Filtro re
    ORDER BY re.rid ASC
),
Relevancia AS (
    SELECT DISTINCT re.rid, re.fid, re.ptext
    , COUNT(re.rid) OVER(PARTITION BY re.rid ) AS relevancia_resposta
    FROM Relevanciaresposta re
    GROUP BY re.rid, re.ptext
    ORDER BY re.rid ASC
),
probabilidade AS (
    SELECT fi.rid,fi.fid
    , MAX(rele.relevancia_resposta) AS relevancia_resposta
    , COUNT(fi.fid) OVER(PARTITION BY fi.rid ) AS qtd_frase
    , MAX(rr.relevancia_frase) AS relevancia_frase
    --, GROUP_CONCAT(p2.ptext, ' ') OVER (PARTITION BY  p2.fid ORDER BY p2.fid ASC, p2.pid ASC)     AS pergunta_correta
    FROM Relevancia rele
    INNER JOIN Filtro fi ON(rele.rid = fi.rid)
    INNER JOIN pergunta p2 ON(fi.fid = p2.fid)
    INNER JOIN Relevanciaresposta rr ON(rele.rid = rr.rid)
    INNER JOIN EntradaOrdenada eo ON(p2.rid = eo.rid AND p2.fid = eo.fid )
    GROUP BY fi.rid,fi.fid
),
perguntando AS (
    SELECT co.relevancia_resposta, co.rid,co.fid, co.qtd_frase, CAST((((co.relevancia_resposta*100.0)/7) ) AS DECIMAL(2,11) ) AS probab, co.relevancia_frase
    FROM probabilidade co
    ORDER BY co.relevancia_resposta DESC
),
pesando AS (
    SELECT DISTINCT co.rid,co.fid, co.qtd_frase, co.relevancia_resposta, co.probab, GROUP_CONCAT(perg.ptext, ' ') OVER (PARTITION BY co.rid,co.fid) AS ptext
    FROM perguntando co
    INNER JOIN pergunta perg ON(co.rid = perg.rid AND co.fid = perg.fid)
    WHERE co.probab >= 68.26
    ORDER BY co.relevancia_resposta DESC
),
Conclusao AS (
    SELECT co.rid,co.fid, co.qtd_frase, co.relevancia_resposta, co.probab, co.ptext, re.rtext
    FROM pesando co
    INNER JOIN resposta re ON(co.rid = re.rid )
)
SELECT co.rid,co.fid, co.qtd_frase, co.relevancia_resposta, co.probab, co.ptext, co.rtext FROM Conclusao co LIMIT 1;
