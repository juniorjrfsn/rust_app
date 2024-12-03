WITH Entrada AS (
    SELECT
        column1 AS id,
        column2 AS ctext,
        column3 AS cord
    FROM (VALUES
        (0,'do',1),
        (0,'Brasil',2),
        (0,'?',3)
    )  AS ptexts
),
EntradaOrdenada AS (
    SELECT DISTINCT conh.id, conh.cid,  E1.cord, E1.ctext,
        ROW_NUMBER()    OVER (PARTITION BY E1.id ORDER BY E1.id ASC, E1.cord ASC)   AS entradaordem,
        MIN(E1.cord)    OVER (PARTITION BY E1.id ORDER BY E1.id ASC )               AS Einicord,
        MAX(E1.cord)    OVER (PARTITION BY E1.id ORDER BY E1.id ASC )               AS Efimcord,
        MIN(conh.id)    OVER (PARTITION BY conh.cid ORDER BY conh.id ASC )          AS Conhinicord,
        MAX(conh.id)    OVER (PARTITION BY conh.cid ORDER BY conh.id ASC )          AS Conhfimcord,
        COUNT(E1.id)    OVER (PARTITION BY E1.id)                                   AS tamanho_entrada,
        COUNT(conh.cid) OVER (PARTITION BY conh.cid)                                AS percentual
    FROM Entrada E1
    INNER JOIN conhecimento conh ON E1.ctext = conh.ctext
    ORDER BY E1.id ASC, conh.cid ASC
),
ConhecimentoCalculado AS (
    SELECT DISTINCT conh.cid, E1.tamanho_entrada,
        COUNT(conh.cid) OVER(PARTITION BY conh.cid) AS conh_tamanho 
        , E1.percentual
    FROM EntradaOrdenada E1
    INNER JOIN conhecimento conh ON( E1.cid = conh.cid )
    GROUP BY conh.id, conh.cid  
    ORDER BY conh.cid ASC 
), -- SELECT * FROM ConhecimentoCalculado
ConhecimentoRelevante AS (
    SELECT conh.cid, COUNT(DISTINCT conh.ctext) AS palavras_encontradas,
        cc.percentual,
        cc.tamanho_entrada
        , COUNT(conh.ctext) OVER (PARTITION BY conh.cid) AS total_palavras_conhecimento
        , CAST(COUNT(DISTINCT conh.ctext) AS FLOAT) / CAST(MAX(E1.tamanho_entrada) AS FLOAT) AS porcentagem_relevancia
        , CAST( ( (cc.percentual * 100.0 / cc.tamanho_entrada) ) AS FLOAT) AS porcentual
    FROM EntradaOrdenada E1
    INNER JOIN conhecimento conh ON E1.ctext = conh.ctext
    INNER JOIN ConhecimentoCalculado cc ON conh.cid = cc.cid
    GROUP BY conh.cid
    ORDER BY palavras_encontradas DESC, porcentagem_relevancia DESC LIMIT 1
), -- SELECT * FROM ConhecimentoRelevante
FraseCompleta AS (
    SELECT DISTINCT conh.id, conh.cid, conh.cord, conh.ctext 
    FROM conhecimento conh
    INNER JOIN ConhecimentoRelevante cr ON(conh.cid = cr.cid ) -- WHERE conh.cid = (SELECT cr.cid FROM ConhecimentoRelevante cr)
), --SELECT * FROM FraseCompleta
PerguntaEncontrada AS(
    SELECT conh.cid, conh.cord, conh.ctext, COUNT(conh.cid) OVER (PARTITION BY conh.cid) AS tamanho_pergunta
    FROM FraseCompleta conh
    WHERE conh.id <= (
        SELECT MAX(Conhfimcord)
        FROM EntradaOrdenada
    )  OR conh.ctext IN (SELECT ctext FROM EntradaOrdenada)
)
  -- SELECT * FROM PerguntaEncontrada
, -- SELECT * FROM PerguntaEncontrada
RespostaEncontrada AS(
    SELECT conh.id, conh.cid, conh.cord, conh.ctext
    FROM FraseCompleta conh
     WHERE conh.id > (SELECT MAX(eo.Conhfimcord) FROM EntradaOrdenada eo)
     -- WHERE conh.cord >= (SELECT MAX(entradaordem) FROM EntradaOrdenada)
    ORDER BY conh.cord
),
Pergunta AS(
    SELECT conh.cid, REPLACE(GROUP_CONCAT(conh.ctext, ' '),'?', '') AS premissa  
    ,cr.tamanho_entrada,cr.porcentual, cr.percentual, conh.tamanho_pergunta , CAST( ( (cr.percentual * 100.0 / conh.tamanho_pergunta) ) AS FLOAT) AS relevancia
    FROM PerguntaEncontrada conh
    INNER JOIN ConhecimentoRelevante cr ON(conh.cid = cr.cid)
)

-- SELECT * FROM Pergunta

SELECT CAST(p.premissa AS TEXT)+ CAST(' ?' AS TEXT) AS premissa, REPLACE(GROUP_CONCAT(re.ctext, ' '),'?', '') AS conclusao
FROM Pergunta p
INNER JOIN RespostaEncontrada re ON(p.cid = re.cid)
-- WHERE relevancia > 19.74

-- SELECT * FROM RespostaEncontrada
