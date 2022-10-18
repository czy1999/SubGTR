import pickle
import random
import torch
import numpy as np
from tcomplex import TComplEx

from tqdm import tqdm


def loadTkbcModel(tkbc_model_file):
    print('Loading tkbc model from', tkbc_model_file)
    x = torch.load(tkbc_model_file, map_location=torch.device("cpu"))
    num_ent = x['embeddings.0.weight'].shape[0]
    num_rel = x['embeddings.1.weight'].shape[0]
    num_ts = x['embeddings.2.weight'].shape[0]
    print('Number ent,rel,ts from loaded model:', num_ent, num_rel, num_ts)
    sizes = [num_ent, num_rel, num_ent, num_ts]
    rank = x['embeddings.0.weight'].shape[1] // 2  # complex has 2*rank embedding size
    tkbc_model = TComplEx(sizes, rank, no_time_emb=False)
    tkbc_model.load_state_dict(x)
    tkbc_model.cuda()
    print('Loaded tkbc model')
    return tkbc_model


def loadTkbcModel_complex(tkbc_model_file):
    print('Loading complex tkbc model from', tkbc_model_file)
    tcomplex_file = 'models/wikidata_big/kg_embeddings/tcomplex_17dec.ckpt'  # TODO: hack
    tcomplex_params = torch.load(tcomplex_file)
    complex_params = torch.load(tkbc_model_file)
    num_ent = tcomplex_params['embeddings.0.weight'].shape[0]
    num_rel = tcomplex_params['embeddings.1.weight'].shape[0]
    num_ts = tcomplex_params['embeddings.2.weight'].shape[0]
    print('Number ent,rel,ts from loaded model:', num_ent, num_rel, num_ts)
    sizes = [num_ent, num_rel, num_ent, num_ts]
    rank = tcomplex_params['embeddings.0.weight'].shape[1] // 2  # complex has 2*rank embedding size

    # now put complex params in tcomplex model

    tcomplex_params['embeddings.0.weight'] = complex_params['embeddings.0.weight']
    tcomplex_params['embeddings.1.weight'] = complex_params['embeddings.1.weight']
    torch.nn.init.xavier_uniform_(tcomplex_params['embeddings.2.weight'])  # randomize time embeddings

    tkbc_model = TComplEx(sizes, rank, no_time_emb=False)
    tkbc_model.load_state_dict(tcomplex_params)
    tkbc_model.cuda()
    print('Loaded complex tkbc model')
    return tkbc_model


def loadTmodel(tkbc_model_file):
    print('Loading tkbc model from', tkbc_model_file)
    x = torch.load(tkbc_model_file, map_location=torch.device("cpu"))
    num_ent = x['embeddings.0.weight'].shape[0]
    num_rel = x['embeddings.1.weight'].shape[0]
    num_ts = x['embeddings.2.weight'].shape[0]
    print('Number ent,rel,ts from loaded model:', num_ent, num_rel, num_ts)
    sizes = [num_ent, num_rel, num_ent, num_ts]
    rank = x['embeddings.0.weight'].shape[1] // 2  # complex has 2*rank embedding size
    tkbc_model = Tmodel(sizes, rank, no_time_emb=False)
    tkbc_model.load_state_dict(x)
    tkbc_model.cuda()
    print('Loaded tkbc model')
    return tkbc_model


def loadRmodel(tkbc_model_file):
    print('Loading tkbc model from', tkbc_model_file)
    x = torch.load(tkbc_model_file, map_location=torch.device("cpu"))
    num_ent = x['embeddings.0.weight'].shape[0]
    num_rel = x['embeddings.1.weight'].shape[0]
    num_ts = x['embeddings.2.weight'].shape[0]
    print('Number ent,rel,ts from loaded model:', num_ent, num_rel, num_ts)
    sizes = [num_ent, num_rel, num_ent, num_ts]
    rank = x['embeddings.0.weight'].shape[1] // 2  # complex has 2*rank embedding size
    tkbc_model = Rmodel(sizes, rank, no_time_emb=False)
    tkbc_model.load_state_dict(x)
    tkbc_model.cuda()
    print('Loaded tkbc model')
    return tkbc_model


def dataIdsToLiterals(d, all_dicts):
    new_datapoint = []
    id2rel = all_dicts['id2rel']
    id2ent = all_dicts['id2ent']
    id2ts = all_dicts['id2ts']
    wd_id_to_text = all_dicts['wd_id_to_text']
    new_datapoint.append(wd_id_to_text[id2ent[d[0]]])
    new_datapoint.append(wd_id_to_text[id2rel[d[1]]])
    new_datapoint.append(wd_id_to_text[id2ent[d[2]]])
    new_datapoint.append(id2ts[d[3]])
    new_datapoint.append(id2ts[d[4]])
    return new_datapoint


def getAllDicts(dataset_name):
    # base_path = '/scratche/home/apoorv/tkbc/tkbc_env/lib/python3.7/site-packages/tkbc-0.0.0-py3.7.egg/tkbc/data/wikidata_small/'
    base_path = 'data/{dataset_name}/kg/tkbc_processed_data/{dataset_name}/'.format(
        dataset_name=dataset_name
    )
    dicts = {}
    for f in ['ent_id', 'rel_id', 'ts_id']:
        in_file = open(str(base_path + f), 'rb')
        dicts[f] = pickle.load(in_file)
    rel2id = dicts['rel_id']
    ent2id = dicts['ent_id']
    ts2id = dicts['ts_id']
    file_ent = 'data/{dataset_name}/kg/wd_id2entity_text.txt'.format(
        dataset_name=dataset_name
    )
    file_rel = 'data/{dataset_name}/kg/wd_id2relation_text.txt'.format(
        dataset_name=dataset_name
    )

    def readDict(filename):
        f = open(filename, 'r')
        d = {}
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 1:
                line.append('')  # in case literal was blank or whitespace
            d[line[0]] = line[1]
        f.close()
        return d

    e = readDict(file_ent)
    r = readDict(file_rel)
    wd_id_to_text = dict(list(e.items()) + list(r.items()))

    def getReverseDict(d):
        return {value: key for key, value in d.items()}

    id2rel = getReverseDict(rel2id)
    id2ent = getReverseDict(ent2id)
    id2ts = getReverseDict(ts2id)

    all_dicts = {'rel2id': rel2id,
                 'id2rel': id2rel,
                 'ent2id': ent2id,
                 'id2ent': id2ent,
                 'ts2id': ts2id,
                 'id2ts': id2ts,
                 'wd_id_to_text': wd_id_to_text
                 }

    return all_dicts


def checkQuestion(question, target_type):
    question_type = question['type']
    if target_type != question_type:
        return False
    return True


# def getDataPoint(question, all_dicts):

def predictTime(question, model, all_dicts, k=1):
    entities = list(question['entities'])
    times = question['times']
    target_type = 'simple_time'
    if checkQuestion(question, target_type) == False:
        print('Not Entity question')
        return set()
    ent2id = all_dicts['ent2id']
    rel2id = all_dicts['rel2id']
    id2ts = all_dicts['id2ts']
    annotation = question['annotation']
    head = ent2id[annotation['head']]
    tail = ent2id[annotation['tail']]
    # relation = rel2id[list(question['relations'])[0]]
    relation = list(question['relations'])[0]
    if 'P' not in relation:
        relation = 'P' + relation
    relation = rel2id[relation]  # + model.embeddings[1].weight.shape[0]//2 #+ 90
    data_point = [head, relation, tail, 1, 1]
    data_batch = torch.from_numpy(np.array([data_point])).cuda()
    time_scores = model.forward_over_time(data_batch)
    val, ind = torch.topk(time_scores, k, dim=1)
    topk_set = set()
    for row in ind:
        for x in row:
            topk_set.add(id2ts[x.item()][0])
    return topk_set


def predictTail(question, model, all_dicts, k=1):
    entities = list(question['entities'])
    times = list(question['times'])
    target_type = 'simple_entity'
    if checkQuestion(question, target_type) == False:
        print('Not Entity question')
        return set()
    ent2id = all_dicts['ent2id']
    rel2id = all_dicts['rel2id']
    ts2id = all_dicts['ts2id']
    id2ent = all_dicts['id2ent']
    head = ent2id[entities[0]]
    try:
        time = ts2id[(times[0], 0, 0)]
    except:
        return set()
    relation = list(question['relations'])[0]
    if 'P' not in relation:
        relation = 'P' + relation
    relation = rel2id[relation]  # + model.embeddings[1].weight.shape[0]//2 #+ 90
    data_point = [head, relation, 1, time, time]
    data_batch = torch.from_numpy(np.array([data_point])).cuda()
    predictions, factors, time = model.forward(data_batch)
    val, ind = torch.topk(predictions, k, dim=1)
    topk_set = set()
    for row in ind:
        for x in row:
            topk_set.add(id2ent[x.item()])
    return topk_set


def checkIfTkbcEmbeddingsTrained(tkbc_model, dataset_name, split='test'):
    filename = 'data/{dataset_name}/questions/{split}.pickle'.format(
        dataset_name=dataset_name,
        split=split
    )
    questions = pickle.load(open(filename, 'rb'))
    all_dicts = getAllDicts(dataset_name)
    for question_type in ['simple_entity', 'simple_time']:
        correct_count = 0
        total_count = 0
        k = 1  # hit at k
        for i in tqdm(range(len(questions))):
            this_question_type = questions[i]['type']
            if question_type == this_question_type and question_type == 'simple_entity':
                which_question_function = predictTail
            elif question_type == this_question_type and question_type == 'simple_time':
                which_question_function = predictTime
            else:
                continue
            total_count += 1
            id = i
            predicted = which_question_function(questions[id], tkbc_model, all_dicts, k)
            intersection_set = set(questions[id]['answers']).intersection(predicted)
            if len(intersection_set) > 0:
                correct_count += 1

        print(question_type, correct_count, total_count, correct_count / total_count)


with open('./saved_pkl/e2rt.pkl', 'rb') as f:
    e2rt = pickle.load(f)
with open('./saved_pkl/event2time.pkl', 'rb') as f:
    event2time = pickle.load(f)
with open('./saved_pkl/e2tr.pkl', 'rb') as f:
    e2tr = pickle.load(f)


def get_neighbours(e):
    tr = e2tr[e]
    neighbours = []
    for t in tr:
        neighbours.append(t[0])
        neighbours.append(t[2])
    neighbours = set(neighbours)
    neighbours.remove(e)
    return list(neighbours)


def check_triples(q1, q2):
    l = []
    if type(q2) == list:
        for i in q2:
            l += check_triples(q1, i)
    else:
        for i in e2rt[(q1, q2)]:
            l.append(i)
        for i in e2rt[(q2, q1)]:
            l.append(i)
    return l


def rerank_ba(pred, entity_in_query, q_type, dataset):
    vs = []
    for p in pred:
        related_triples = check_triples(p, list(entity_in_query))
        v = 9999
        if p in entity_in_query or len(related_triples) == 0:
            pass
        elif q_type == 'before':
            for t in related_triples:
                deta = dataset[7] - int(t[3]) - 1
                if deta < v and deta >= 0:
                    v = deta
        else:
            for t in related_triples:
                deta = int(t[4]) - dataset[8] - 1
                if deta < v and deta >= 0:
                    v = deta
        vs.append(v)
    value_rank = np.argsort(vs)

    return [pred[ix] for ix in value_rank]


def rerank_fl(pred, entity_in_query, q_type, dataset):
    vs = []
    for p in pred:
        related_triples = check_triples(p, list(entity_in_query))
        v = 9999
        if p in entity_in_query or len(related_triples) == 0:
            pass
        elif q_type == 'first':
            for t in related_triples:
                deta = int(t[3])
                if deta < v:
                    v = deta
        else:
            for t in related_triples:
                deta = 9500 - int(t[4])
                if deta < v and deta >= 0:
                    v = deta
        vs.append(v)
    value_rank = np.argsort(vs)

    return [pred[ix] for ix in value_rank]


def rerank_st(pred, entity_in_query, q_type, dataset, event=False):
    vs = []
    ans = []
    if len(entity_in_query) == 2:
        trs = check_triples(list(entity_in_query)[0], list(entity_in_query)[1])
        for t in trs:
            ans.append(int(t[3]))
            pred.append(int(t[3]))
    for p in pred:
        v = 9999
        if p in ans:
            v = 0
        vs.append(v)
    value_rank = np.argsort(vs)

    return [pred[ix] for ix in value_rank]


def rerank_tj(pred, entity_in_query, q_type, dataset, event=False):
    vs = []
    for p in pred:
        related_triples = check_triples(p, list(entity_in_query))
        v = 9999

        if p not in entity_in_query and len(related_triples) != 0:
            for tp in related_triples:
                start = int(tp[3])
                end = int(tp[4])
                if not (start > dataset[8] or end < dataset[7]):
                    v = 0
        vs.append(v)
    value_rank = np.argsort(vs)

    return [pred[ix] for ix in value_rank]


def subgraph_reasoning(dataset, index, question, predicted):
    i = index
    neighbours = []
    # # ADD subgraph neighbours
    # for e in question['entities']:
    #     neighbours += get_neighbours(e)
    # for neighbour in neighbours:
    #     predicted.append(neighbour)
    if question['type'] == 'before_after':
        if 'event_head' in question['annotation'].keys():
            event = question['annotation']['event_head']
            if event[0] != 'Q':
                t = int(event)
            else:
                t = int(list(event2time[event])[0][3])
            d = list(dataset[i])
            d[7] = t
            d[8] = t
            predicted = rerank_ba(predicted, question['entities'], question['annotation']['type'], d)[:k]
        else:
            predicted = rerank_ba(predicted, question['entities'], question['annotation']['type'],
                                  dataset[i])

    if question['type'] == 'first_last':
        if question['answer_type'] == 'entity':
            predicted = rerank_fl(predicted, question['entities'], question['annotation']['adj'],
                                  dataset[i])

    if question['type'] == 'simple_time':
        predicted = rerank_st(predicted, question['entities'], None,
                              dataset[i])

    if question['type'] == 'time_join':
        if len(question['entities']) == 2:
            if 'event_head' in question['annotation'].keys():
                event = question['annotation']['event_head']
                if event[0] != 'Q':
                    t = int(event)
                else:
                    t = int(list(event2time[event])[0][3])
                d = list(dataset[i])
                d[7] = t
                d[8] = t
                predicted = rerank_tj(predicted, question['entities'], None, d)[:k]
            else:
                predicted = rerank_tj(predicted, question['entities'], None,
                                      dataset[i])
    return predicted


def print_info(args):
    print('#######################')
    print('Model: ' + args.model)
    print('QA Dataset: ' + args.dataset_name)
    print('Aware Module: ' + str(args.aware_module))
    print('Subgraph Reasoning: ' + str(args.subgraph_reasoning))
    print('Time Sensitivity: ' + str(args.time_sensitivity))
    print('TKG Embeddings: ' + args.tkbc_model_file)
    print('TKG for QA (if applicable): ' + args.tkg_file)
    print('#######################')
