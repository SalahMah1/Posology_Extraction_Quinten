import json
import numpy as np
import pandas as pd
from random import randint
import random
import re
from itertools import compress
from posology_extraction_gr8.src.DataAugmentation.inventedMedicalDatabase import texteMedicalInvente

def load_train_original_df(url):
    
    with open(url, 'r') as json_f:
        json_list = list(json_f)

    id = np.array([])
    text = np.array([])

    label = []
    comments = []

    for json_str in json_list:
        
        result = json.loads(json_str)
        id = np.append(id, result['id'])
        text = np.append(text, result['text'])
        label.append(result['label'])
        comments.append(result['Comments'])

    df = pd.DataFrame([id, text, label, comments]).T
    df.columns = ['id', 'text', 'labels', 'comments']
    df['num_labels'] = [len(ll) for ll in df['labels']]

    return df

def extract_labels(df):

    k = 0
    word_extracted = []
    label_extracted = []

    for l in df['labels']:

        for q in range(len(l)):
            
            start = l[q][0]
            end = l[q][1]
            label_o = l[q][2]
            word_o = df['text'][k][start:end]
            word_extracted.append(word_o)
            label_extracted.append(label_o)

        k +=1

    df_labels = pd.DataFrame({'expression': word_extracted, 'label': label_extracted})

    return df_labels

def randomize_label(df_labels, type_label):
    # Dosage
    dosage_met = ['mL/h', 'kg', 'µg', 'mg', 'g', 'ng/m', 'kg/m2', 'µg/m2', 'mg/m2', 'g/m2', 'ng/m2', 'kg/m3', 'µg/m3', 'mg/m3', 'g/m3', 
        'ng/m3', 'kg/L', 'µg/L', 'mg/L', 'g/L', 'ng/L', 'kg/l', 'µg/l', 'mg/l', 'g/l', 'ng/l', 'kg/ml', 'µg/ml', 'mg/ml', 
        'g/ml', 'ng/ml', 'RNI', 'm', '%', 'µg', 'perfusions', 'ampoules', 'cps', 'mg/Kg/Jou', 'flacon', 'MUI', 'unités', 
        'g/kg', 'µg/kg', 'mmol/L', 'L', 'ml', 'gy', 'Gray', 'comprimés', 'UI/kg', 'mg/m²', 'cp', 'micromole', 'gramme', 
        'kilogramme', 'µg/kg', 'inj', 'GHB', 'mEq', 'cycles', 'instillations', 'dose ', 'milligrammes', 'sachet', 'sachets', 
        'cuillères', 'UI', 'boites']
    dosage_val = randint(0, 5639)/100
    dosage_fin = str(dosage_val) + ' ' + random.choice(dosage_met)

    # Drug
    list_extern_drug = list((pd.read_csv('posology_extraction_gr8/Inputs/vidal_substances.csv'))['substance_name'])
    list_extern_drug_sacha = list(set((pd.read_csv('posology_extraction_gr8/Inputs/drug_database.csv'))['drug']))
    drug_list = list(set(df_labels[df_labels['label'] == 'DRUG'].expression)) + list_extern_drug + list_extern_drug_sacha
    drug_list.remove('\n+ p')
    drug_list.remove('D')
    drug_list.remove('20 mg')
    drug_list.remove(' LSD')
    drug_list.remove('C')

    # Frequency
    x = randint(1, 5)
    frequency_list = [' au besoin', ' '+str(x)+' fois par jour', ' '+str(x)+' par jour', ' '+str(x)+'/j',  
        ' '+str(x)+'/jour',  ' '+str(x)+' fois par jour matin et soir', ' '+str(x)+'/24 h', ' '+str(x)+' quotidien', 
         ' '+str(x)+' toutes les 12 heures',  ' '+str(x)+' x/jour',  ' '+str(x)+' séances par semaine']

    # Treatment
    treatment_list = list(set(df_labels[df_labels['label'] == 'TREATMENT'].expression))

    # Route
    list_extern_route_sacha = list(set((pd.read_csv('posology_extraction_gr8/Inputs/drug_database.csv'))['route']))
    route_list = list(set(df_labels[df_labels['label'] == 'ROUTE'].expression)) + list_extern_route_sacha

    # Duration
    duration_val = randint(1, 30)
    duration_met = [' pendant ' + str(duration_val) + ' jours', ' pendant ' + str(duration_val) + ' semaines', ' pendant ' + str(duration_val) + ' mois', 
        ' pendant les ' + str(duration_val) + ' prochaines mois', ' pendant deux jours', ' pendant trois jours', ' pendant quatres jours', 
        ' pendant cinq jours', ' pendant six jours', ' pendant sept jours', ' pendant huit jours', ' pendant neuf jours', 
        ' pendant dix jours', ' pendant quatorze jours', ' pour les ' + str(duration_val) + ' prochaines semaines', 
        ' pour les ' + str(duration_val) + ' prochains mois', ' pour les ' + str(duration_val) + ' prochains jours', ' pour les deux prochaines semaines', 
        ' pour les trois prochaines semaines', ' pour les quatres prochaines semaines', ' pour les cinq prochaines semaines', 
        ' pour les six prochaines semaines', ' pour les sept prochaines semaines', ' pour les huit prochaines semaines', 
        ' pour les neuf prochaines semaines', ' pour les dix prochaines semaines', ' pour les deux prochains mois', 
        ' pour les trois prochains mois', ' pour les quatres prochains mois', ' pour les cinq prochains mois', 
        ' pour les six prochains mois', ' pour les sept prochains mois', ' pour les huit prochains mois', 
        ' pour les neuf prochains mois', ' pour les dix prochains mois', ' pour les deux prochains jours', 
        ' pour les trois prochains jours', ' pour les quatres prochains jours', ' pour les cinq prochains jours', 
        ' pour les six prochains jours', ' pour les sept prochains jours', ' pour les huit prochains jours', 
        ' pour les neuf prochains jours', ' pour les dix prochains jours', ' pour deux semaines', ' pour trois semaines', 
        ' pour quatres semaines', ' pour cinq semaines', ' pour six semaines', ' pour sept semaines', ' pour huit semaines', 
        ' pour neuf semaines', ' pour dix semaines', ' pour deux mois', ' pour trois mois', ' pour quatres mois', ' pour cinq mois', 
        ' pour six mois', ' pour sept mois', ' pour huit mois', ' pour neuf mois', ' pour dix mois', ' pour deux jours', 
        ' pour trois jours', ' pour quatres jours', ' pour cinq jours', ' pour six jours', ' pour sept jours', ' pour huit jours', 
        ' pour neuf jours', ' pour dix jours', ' sur deux semaines', ' sur trois semaines', ' sur quatres semaines', 
        ' sur cinq semaines', ' sur six semaines', ' sur sept semaines', ' sur huit semaines', ' sur neuf semaines', 
        ' sur dix semaines', ' sur deux mois', ' sur trois mois', ' sur quatres mois', ' sur cinq mois', ' sur six mois', 
        ' sur sept mois', ' sur huit mois', ' sur neuf mois', ' sur dix mois', ' sur deux jours', ' sur trois jours', 
        ' sur quatres jours', ' sur cinq jours', ' sur six jours', ' sur sept jours', ' sur huit jours', ' sur neuf jours', 
        ' sur dix jours', " J1 jusqu’à un arrêt à J " + str(duration_val), ' à vie']

    # Form
    list_extern_form_sacha = list(set((pd.read_csv('posology_extraction_gr8/Inputs/drug_database.csv'))['form']))
    form_list = list(set(df_labels[df_labels['label'] == 'FORM'].expression)) + list_extern_form_sacha

    if type_label=='DOSAGE':
        r_label = dosage_fin
    elif type_label=='DRUG':
        r_label = random.choice(drug_list)
    elif type_label=='FREQUENCY':
        r_label = random.choice(frequency_list)
    elif type_label=='TREATMENT':
        r_label = random.choice(treatment_list)
    elif type_label=='ROUTE':
        r_label = random.choice(route_list)
    elif type_label=='DURATION':
        r_label = random.choice(duration_met)
    else:
        r_label = random.choice(form_list)

    r_label = ' ' + str(r_label) + ' '
    return r_label

def relocate_label(lab_lst, origin_len):
    
    
    label_list = []
    start_label_lst = []
    end_label_lst = []
    old_start_label_lst = []
    old_end_label_lst = []
    sent_loc_lst = []

    for q in range(len(lab_lst)):
        start = lab_lst[q][0]
        end = lab_lst[q][1]
        label_o = lab_lst[q][2]
        for k in range(len(origin_len)):
            if k==0:
                if start<origin_len[k]:
                    new_start = start-1
                    new_end = new_start + end - start
                    sentence_loc = k+1
                    break
            else:
                if start<origin_len[k]:
                    new_start = start - origin_len[k-1]-1
                    new_end = new_start + end - start
                    sentence_loc = k+1
                    break
        
        label_list.append(label_o)
        start_label_lst.append(new_start)
        end_label_lst.append(new_end)
        old_start_label_lst.append(start)
        old_end_label_lst.append(end)
        sent_loc_lst.append(sentence_loc)

    return label_list, start_label_lst, end_label_lst, old_start_label_lst, old_end_label_lst, sent_loc_lst

def extract_sentences(df):
    
    df_aug = df[['id', 'labels', 'text']].copy()
    # df_aug['txt_split'] = df['text'].str.split('.')
    df_aug['txt_split'] = [re.split(r'(?<!\d)\.(?!\d|$)', l_txt) for l_txt in df['text']]
    text_lst = []
    text_len_lst = []
    for i in range(len(df_aug)):
        text_lst.append(len(df_aug['txt_split'][i]))
        to_append_int_lst = [len(t) for t in df_aug['txt_split'][i]]
        to_append_lst = [(ta+1) if (to_append_int_lst.index(ta)!=0) else ta for ta in to_append_int_lst]
        text_len_lst.append(to_append_lst)
    df_aug['text_lst'] = text_lst
    df_aug['text_len_lst'] = text_len_lst
    cumsum_len = list([np.cumsum(cs) for cs in df_aug['text_len_lst']])

    df_aug['origin_len'] = cumsum_len

    df_aug[['label_list', 'start_label_lst', 'end_label_lst', 'old_start_label_lst', 'old_end_label_lst', 
        'sent_loc_lst']] = [relocate_label(df_aug['labels'][i], df_aug['origin_len'][i]) for i in range(len(df_aug))]
        
    df_aug = df_aug.explode('txt_split').reset_index(drop=True)

    df_aug = df_aug[df_aug['labels'].str.len()!=0].reset_index(drop=True)

    sentence_num = [1]
    stay = [False]
    k = 1
    for i in range(1, len(df_aug)):
        if df_aug['id'][i]==df_aug['id'][i-1]:
            k += 1
        else:
            k = 1
        stay.append((k in df_aug['sent_loc_lst'][i]))
        sentence_num.append(k)
    df_aug['sentence_num'] = sentence_num
    df_aug['stay'] = stay
    df_aug = df_aug[df_aug['stay'] == True].reset_index(drop=True)

    df_aug_txt = df_aug[['id', 'txt_split', 'label_list', 'start_label_lst', 'end_label_lst', 'sentence_num', 'sent_loc_lst']].copy()
    final_lab_list = []
    for i in range(len(df_aug_txt)):

        mask = [df_aug_txt['sentence_num'][i]==qq for qq in df_aug_txt['sent_loc_lst'][i]]

        lab_l = df_aug_txt['label_list'][i]
        start_l = df_aug_txt['start_label_lst'][i]
        end_l = df_aug_txt['end_label_lst'][i]

        df_aug_txt['label_list'][i] = list(compress(lab_l, mask))
        df_aug_txt['start_label_lst'][i] = list(compress(start_l, mask))
        df_aug_txt['end_label_lst'][i] = list(compress(end_l, mask))

        lst_int = []
        for j in range(len(df_aug_txt['label_list'][i])):
            lst_int.append([df_aug_txt['start_label_lst'][i][j], df_aug_txt['end_label_lst'][i][j], df_aug_txt['label_list'][i][j]])
        
        final_lab_list.append(lst_int)

    df_aug_txt['final_lab'] = final_lab_list


    df_final = df_aug_txt[['txt_split', 'final_lab']].copy()

    return df_final

def create_aug(df, df_labels, num_sent_used=125, num_sample_per=3):
    
    lst_selected = random.sample(range(len(df)-1), num_sent_used)
    df_s = df[df.index.isin(lst_selected)]
    df_s = pd.concat([df_s]*num_sample_per, ignore_index=True)

    n_w = []
    n_l = []
    n_s = []
    n_e = []
    l_l = []
    n_lab_f = []

    for i in range(len(df_s)):
        start_l = []
        end_l = []
        lab_l = []
        len_l = []
        word = []
        new_word = []
        new_start = []
        new_end = []
        lst_int = []
        words_k = []
        k_ind = 0

        for q in df_s['final_lab'][i]:          
            words_k.append(df_s['txt_split'][i][q[0]:q[1]])
        
        for k in df_s['final_lab'][i]:

            word_k = words_k[k_ind]
            lab_rep = k[2]
            new_word = randomize_label(df_labels, lab_rep)
            df_s['txt_split'][i] = df_s['txt_split'][i].replace(word_k, new_word)
            
            new_start_k = df_s['txt_split'][i].find(new_word)

            new_end_k = new_start_k + len(new_word)

            word.append(new_word)
            lab_l.append(lab_rep)
            len_l.append(len(new_word))
            new_start.append(new_start_k)
            new_end.append(new_end_k)

            lst_int.append([new_start_k, new_end_k, lab_rep])

            k_ind += 1

        n_w.append(word)
        n_l.append(lab_l)
        l_l.append(len_l)
        n_s.append(new_start)
        n_e.append(new_end)

        n_lab_f.append(lst_int)

    df_s['labels'] = n_lab_f
    df_s.drop('final_lab', axis=1, inplace=True)

    df_s.rename(columns={"txt_split": "text", "labels": "labels"}, inplace=True)

    return df_s

def fonction_replace(texte, df_labels):
    '''
    replaces brakets in text (string) by random occurences of labels in created text.
    for instance : 
    In : "la femme a reçus [dosage] [drug] par [route]."
    Out : "la femme a reçus 3 ampoules doliprane par voie orale" + la position des labels dans le texte 
    
    attention : the structure of the sentence is correct but has no medical accuracy.
    '''

    label = []
    nb_labels = len(re.findall(r"\[(\w+)\]",texte))
    current = nb_labels
    
    while current!= 0:#goes 1 by 1
        m = re.search(r"\[(\w+)\]", texte)##searches for brackets 
        start = m.start()##gets the position of the token 
        
        lab_interest = (m.group())[1:-1]
        lab_interest_bracket = (m.group())
        current_label = lab_interest.upper()
        if current_label in ['TREATMENT', 'FREQUENCY', 'DURATION', 'DOSAGE', 'DRUG', 'ROUTE', 'FORM']:
            form_random = randomize_label(df_labels, current_label)
            chaine = texte.replace(lab_interest_bracket, form_random, 1)# trouve [form] à remplacer dans le texte 
            longeur = len(form_random)-2 # has to take into account the brakets 
            end = start+longeur+1
            label.append([start, end, current_label]) # ajoute la position du label dans le texte
        else:
            pass
        texte = chaine[:]
        #print(current)
        current -=1

    return label, texte

def generate_generic_str(liste_texte, df_labels):
    '''
    in : takes a list of generated texts with gaps
    out : returns a dataframe with filled gaps and the according labels.    
    makes a copy of the dataframe in csv format 
    '''

    # PATH_output = r"posology_extraction_gr8/Inputs/augmentated_data_generated.csv"
    liste_label = []
    liste_nv_texte = []
    for texte in liste_texte:
        labels,nv_texte = fonction_replace(texte, df_labels)
        liste_label.append(labels)
        liste_nv_texte.append(nv_texte)

    d = {'text':liste_nv_texte,'list_label':liste_label}
    df = pd.DataFrame(data=d) # create dataframe from dictionnart

    df.rename(columns={"text": "text", "list_label": "labels"}, inplace=True)

    # df.to_csv(PATH_output)
    return df

def augment_data(num_sent_used=150, num_sample_per=3, merge_ds=True):

    url = 'posology_extraction_gr8/src/DataAugmentation/data_training_labeled.jsonl'
    df = load_train_original_df(url)
    df_labels = extract_labels(df)
    df_final = extract_sentences(df)
    df_augmented = create_aug(df_final, df_labels, num_sent_used, num_sample_per)

    if merge_ds==True:
        df_generic_augmented = generate_generic_str(texteMedicalInvente, df_labels)
        df_augmented.append(df_generic_augmented, ignore_index=True)

    df_augmented

    return df_augmented