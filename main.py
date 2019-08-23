from dwpi_process import DataLoad, ExtractFields, CleanFields, ProcessDwpiText, ComputeDwpi, CreateMatrix
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
if __name__ == '__main__':
    load = DataLoad(r'F:\project\pycharm_workplace\dwpi\data')
    records = load.getAllRecords()
    croups = list()
    for record in records:
        fields_ti_ab = ExtractFields.extract_fields(record, "TI", "AB")
        fields_ti_ab["AB"] = CleanFields.clean_ab(fields_ti_ab["AB"])
        dwpi_text = ProcessDwpiText()
        tokenize_ab_ti = dwpi_text.word_from_list_tokenize(fields_ti_ab)
        grammar = r"""
                    NBAR:
                        {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

                    NP:
                        {<NBAR>}
                        {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...;IN代表介词
                """
        np_leaves = dwpi_text.fields_leaves(tokenize_ab_ti, grammar, "NP")
        croup = dwpi_text.filter_normalise_word(np_leaves)
        croup = itertools.chain.from_iterable([croup['TI'], croup['AB']])   #合并所有列表中的单词到同一个列表中作为一个文档
        croups.append(list(croup))

    keyword_list = ComputeDwpi.get_keyword_list(croups)
    matrix = CreateMatrix()
    tf_idf_matrix = matrix.tf_idf(keyword_list, croups)
    similarity = cosine_similarity(tf_idf_matrix)
    df = pd.DataFrame(similarity)
    df.to_excel('./similarity.xlsx')
    print('finally')

