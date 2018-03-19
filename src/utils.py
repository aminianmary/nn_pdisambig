from collections import Counter, defaultdict
import re, codecs, random
import numpy as np

class ConllStruct:
    def __init__(self, entries, predicates):
        self.entries = entries
        self.predicates = predicates

    def __len__(self):
        return len(self.entries)

class ConllEntry:
    def __init__(self, id, form, lemma, pos, sense='_', parent_id=-1, relation='_',  arg_list=dict(),
                 is_pred=False):
        self.id = id
        self.form = form[0:50]
        self.lemma = lemma
        self.norm = normalize(form)[0:50]
        self.lemmaNorm = normalize(lemma)
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.arg_list= arg_list
        self.relation = relation
        self.sense = sense
        self.is_pred = is_pred

    def __str__(self):
        entry_list = [str(self.id+1), self.form, self.lemma, self.lemma, self.pos, self.pos, '_', '_',
                      self.parent_id,
                      self.parent_id, self.relation, self.relation,
                      'Y' if self.is_pred == True else '_',
                      self.sense]
        for p in self.arg_list.values():
            entry_list.append(p)
        return '\t'.join(entry_list)

def vocab(sentences, min_count=2):
    wordsCount = Counter()
    posCount = Counter()
    psenseCount = Counter()
    pWordCount = Counter()
    pLemmaCount = Counter()
    chars = set()

    for sentence in sentences:
        wordsCount.update([node.norm for node in sentence.entries])
        posCount.update([node.pos for node in sentence.entries])
        for node in sentence.entries:
            if node.is_pred:
                pWordCount.update([node.norm])
                pLemmaCount.update([node.lemma])
                psenseCount.update([node.sense])
            for c in list(node.form):
                    chars.add(c.lower())

    words = set()
    for w in wordsCount.keys():
        if wordsCount[w] >= min_count:
            words.add(w)
    pWords = set()
    for l in pWordCount.keys():
        if pWordCount[l] >= min_count:
            pWords.add(l)
    pLemmas = set()
    for l in pLemmaCount.keys():
        if pLemmaCount[l] >= min_count:
            pLemmas.add(l)
    return (list(words), list(pWords), list(pLemmas),
            list(posCount), list(psenseCount.keys()), list(chars))

def sense_mask (sentences, senses, pwords, plemmas, use_lemma):
    senses = ['<UNK>'] + senses
    p = pwords if not use_lemma else plemmas
    p_len = len(pwords)+2 if not use_lemma else len(plemmas)+3
    p_dic = {word: ind + 2 for ind, word in enumerate(p)} if not use_lemma else {word: ind + 3 for ind, word in enumerate(p)}
    sense_dic = {s: ind for ind, s in enumerate(senses)}
    sense_mask = np.full((p_len, len(senses)), -np.inf)
    sense_mask[0] = [0]*len(senses)
    sense_mask[1] = [-np.inf]*len(senses)
    sense_mask[1][0] = np.inf
    for sentence in sentences:
        for node in sentence.entries:
            if node.is_pred:
                word = node.norm if not use_lemma else node.lemma
                w_index = p_dic[word] if word in p_dic else 0
                s_index = sense_dic[node.sense]
                sense_mask[w_index][s_index] = 0
    return sense_mask

def get_predicates_list (sentences, pWords, plemmas, use_lemma, use_default):
    p = []
    for sentence in sentences:
        for node in sentence.entries:
            if node.is_pred:
                p_index = -1
                if use_lemma:
                    lemma = node.lemma
                    p_index = plemmas[lemma] if lemma in plemmas else (1 if use_default else 0)
                else:
                    word = node.norm
                    p_index = pWords[word] if word in pWords else (1 if use_default else 0)
                p.append(p_index)
    return p


def read_conll(fh):
    sentences = codecs.open(fh, 'r').read().strip().split('\n\n')
    read = 0
    for sentence in sentences:
        words = []
        predicates = list()
        entries = sentence.strip().split('\n')
        for entry in entries:
            spl = entry.split('\t')
            arg_list = dict()
            is_pred = False
            if spl[12] == 'Y':
                is_pred = True
                predicates.append(int(spl[0]) - 1)

            for i in range(14, len(spl)):
                arg_list[i - 14] = spl[i]

            words.append(
                ConllEntry(int(spl[0]) - 1, spl[1], spl[3], spl[5], spl[13], spl[9], spl[11], arg_list,
                           is_pred))
        read += 1
        yield ConllStruct(words, predicates)
    print read, 'sentences read.'

def write_conll(fn, conll_structs):
    with codecs.open(fn, 'w') as fh:
        for conll_struct in conll_structs:
            for i in xrange(len(conll_struct.entries)):
                entry = conll_struct.entries[i]
                fh.write(str(entry))
                fh.write('\n')
            fh.write('\n')

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
urlRegex = re.compile("((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)")

def normalize(word):
    return '<NUM>' if numberRegex.match(word) else ('<URL>' if urlRegex.match(word) else word.lower())

def get_batches(buckets, model, is_train, sen_cut):
    d_copy = [buckets[i][:] for i in range(len(buckets))]
    if is_train:
        for dc in d_copy:
            random.shuffle(dc)
    mini_batches = []
    batch, pred_ids, cur_len, cur_c_len = [], [], 0, 0
    b = model.options.batch if is_train else model.options.dev_batch_size
    for dc in d_copy:
        for d in dc:
            if (is_train and len(d)<=sen_cut) or not is_train:
                for p, predicate in enumerate(d.predicates):
                    batch.append(d.entries)
                    pred_ids.append([p,predicate])
                    cur_c_len = max(cur_c_len, max([len(w.norm) for w in d.entries]))
                    cur_len = max(cur_len, len(d))

            if cur_len * len(batch) >= b:
                add_to_minibatch(batch, pred_ids, cur_c_len, cur_len, mini_batches, model)
                batch, pred_ids, cur_len, cur_c_len = [], [], 0, 0

    if len(batch)>0 and not is_train:
        add_to_minibatch(batch, pred_ids, cur_c_len, cur_len, mini_batches, model)
    if is_train:
        random.shuffle(mini_batches)
    return mini_batches


def add_to_minibatch(batch, pred_ids, cur_c_len, cur_len, mini_batches, model):
    words = np.array([np.array(
        [model.words.get(batch[i][j].norm, 0) if j < len(batch[i]) else model.PAD for i in
         range(len(batch))]) for j in range(cur_len)])
    pwords = np.array([np.array(
        	[model.x_pe_dict.get(batch[i][j].norm, 0) if j < len(batch[i]) else model.PAD for i in range(len(batch))])
            for j in range(cur_len)])
    pos = np.array([np.array(
        [model.pos.get(batch[i][j].pos, 0) if j < len(batch[i]) else model.PAD for i in
         range(len(batch))]) for j in range(cur_len)])
    lemmas = np.array([np.array(
        [(model.plemmas.get(batch[i][j].lemma, 0) if pred_ids[i][1] == j else model.NO_LEMMA) if j < len(
            batch[i]) else model.PAD for i in
         range(len(batch))]) for j in range(cur_len)])
    pred_lemmas = np.array([model.plemmas.get(batch[i][pred_ids[i][1]].lemma, 0) for i in range(len(batch))])
    pred_lemmas_index = np.array([pred_ids[i][1] for i in range(len(batch))])
    senses = np.array([np.array(
        [model.senses.get(batch[i][j].sense, 0) if j < len(batch[i]) else 0 for i in
         range(len(batch))]) for j in range(cur_len)])
    chars = np.array([[[model.char_dict.get(batch[i][j].form[c].lower(), 0) if 0 < j < len(batch[i]) and c < len(
        batch[i][j].form) else (1 if j == 0 and c == 0 else 0) for i in range(len(batch))] for j in range(cur_len)] for
                      c in range(cur_c_len)])
    chars = np.transpose(np.reshape(chars, (len(batch) * cur_len, cur_c_len)))
    masks = np.array([np.array([1 if j < len(batch[i]) and batch[i][j].sense !='?' else 0 for i in range(len(batch))]) for j in range(cur_len)])
    mini_batches.append((words, pos, pwords, pos, lemmas, pred_lemmas, pred_lemmas_index, chars, senses, masks))


def get_scores(fp):
    labeled_f = 0
    unlabeled_f = 0
    line_counter =0
    with codecs.open(fp, 'r') as fr:
        for line in fr:
            line_counter+=1
            if line_counter == 10:
                spl = line.strip().split(' ')
                labeled_f= spl[len(spl)-1]
            if line_counter==13:
                spl = line.strip().split(' ')
                unlabeled_f = spl[len(spl) - 1]
    return (labeled_f, unlabeled_f)


def eval_sense(gold_file, predicted_file):
    r1 = codecs.open(gold_file, 'r')
    r2 = codecs.open(predicted_file, 'r')
    l1 = r1.readline()
    c, a_ = 0, 0
    while l1:
        l2 = r2.readline().strip()
        spl = l1.strip().split('\t')
        if len(spl) > 8:
            g_s, p_s = spl[13], l2.split('\t')[13]
            if g_s != '_':
                a_ += 1
                if g_s == p_s:
                    c += 1
        l1 = r1.readline()
    return round(100.0 * float(c)/a_ ,2)