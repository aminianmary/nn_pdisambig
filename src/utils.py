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
    def __init__(self, id, form, lemma, pos, sense='_', parent_id=-1, relation='_', predicateList=dict(),
                 is_pred=False):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.norm = normalize(form)
        self.lemmaNorm = normalize(lemma)
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation
        self.predicateList = predicateList
        self.sense = sense
        self.is_pred = is_pred

    def __str__(self):
        entry_list = [str(self.id+1), self.form, self.lemma, self.lemma, self.pos, self.pos, '_', '_',
                      self.parent_id,
                      self.parent_id, self.relation, self.relation,
                      'Y' if self.is_pred == True else '_',
                      self.sense]
        for p in self.predicateList.values():
            entry_list.append(p)
        return '\t'.join(entry_list)

def vocab(sentences, min_count=2):
    wordsCount = Counter()
    posCount = Counter()
    semRelCount = Counter()
    pWordCount = Counter()
    chars = set()

    for sentence in sentences:
        wordsCount.update([node.norm for node in sentence.entries])
        posCount.update([node.pos for node in sentence.entries])
        for node in sentence.entries:
            if node.predicateList == None:
                continue
            if node.is_pred:
                pWordCount.update([node.norm])
            for pred in node.predicateList.values():
                if pred!='?':
                    semRelCount.update([pred])
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
    return (list(words), list(pWords),
            list(posCount), list(semRelCount.keys()), list(chars))

def read_conll(fh):
    sentences = codecs.open(fh, 'r').read().strip().split('\n\n')
    read = 0
    for sentence in sentences:
        words = []
        predicates = list()
        entries = sentence.strip().split('\n')
        for entry in entries:
            spl = entry.split('\t')
            predicateList = dict()
            is_pred = False
            if spl[12] == 'Y':
                is_pred = True
                predicates.append(int(spl[0]) - 1)

            for i in range(14, len(spl)):
                predicateList[i - 14] = spl[i]

            words.append(
                ConllEntry(int(spl[0]) - 1, spl[1], spl[3], spl[5], spl[13], spl[9], spl[11], predicateList,
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

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

def get_batches(buckets, model, is_train):
    d_copy = [buckets[i][:] for i in range(len(buckets))]
    if is_train:
        for dc in d_copy:
            random.shuffle(dc)
    mini_batches = []
    batch, pred_ids, cur_len, cur_c_len = [], [], 0, 0
    for dc in d_copy:
        for d in dc:
            if (is_train and len(d)<=100) or not is_train:
                for p, predicate in enumerate(d.predicates):
                    batch.append(d.entries)
                    pred_ids.append([p,predicate])
                    cur_c_len = max(cur_c_len, max([len(w.norm) for w in d.entries]))
                    cur_len = max(cur_len, len(d))

            if cur_len * len(batch) >= model.options.batch:
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
    pred_flags = np.array([np.array([(1 if pred_ids[i][1] == j else 0) if j < len(batch[i]) else 0 for i in range(len(batch))]) for j in range(cur_len)])
    pred_lemmas_index = np.array([pred_ids[i][1] for i in range(len(batch))])
    roles = np.array([np.array(
        [model.roles.get(batch[i][j].predicateList[pred_ids[i][0]], 0) if j < len(batch[i]) else model.PAD for i in
         range(len(batch))]) for j in range(cur_len)])
    chars = np.array([[[model.char_dict.get(batch[i][j].form[c].lower(), 0) if 0 < j < len(batch[i]) and c < len(
        batch[i][j].form) else (1 if j == 0 and c == 0 else 0) for i in range(len(batch))] for j in range(cur_len)] for
                      c in range(cur_c_len)])
    chars = np.transpose(np.reshape(chars, (len(batch) * cur_len, cur_c_len)))
    masks = np.array([np.array([1 if j < len(batch[i]) and batch[i][j].predicateList[pred_ids[i][0]]!='?' else 0 for i in range(len(batch))]) for j in range(cur_len)])
    mini_batches.append((words, pos, pwords, pos, pred_lemmas_index, chars, roles, pred_flags, masks))


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