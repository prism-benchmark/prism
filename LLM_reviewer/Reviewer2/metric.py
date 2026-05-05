import gem_metrics
import numpy as np


'''
references: list of references (may be list of lists for multi-ref)
predictions: list of predictions
metrics: list of metrics to compute
'''
def get_results(references, predictions, metrics=['msttr', 'ngrams', 'bleu', 'rouge', 'bertscore']): #meteor

    refs = gem_metrics.texts.References(references)
    preds = gem_metrics.texts.Predictions(predictions)
    return gem_metrics.compute(preds, refs, metrics_list=metrics)


'''
references: list of references (may be list of lists for multi-ref)
predictions: list of predictions
metric: the metric used to compute the similarity in specificity
sub_metric: sub_metric in metric to use compute the similarity in specificity
'''
def get_specificity(references, predictions, metric='bertscore', sub_metric='f1'):

    # random shuffle indices
    self_indices = np.arange(len(predictions))
    indices = np.random.permutation(len(predictions))
    while (self_indices == indices).sum():
        indices = np.random.permutation(len(predictions))

    # shuffled
    shuffle_predictions = np.array(predictions)[indices]
    shuffle_result = get_results(references, shuffle_predictions.tolist(), [metric])

    # original
    result = get_results(references, predictions, [metric])

    # specificity
    spec = result[metric][sub_metric] - shuffle_result[metric][sub_metric]

    return spec


'''
references: list of references (may be list of lists for multi-ref)
predictions: list of predictions (same size as references; may be list of lists for multi-prompt)
metric: the metric used to compute the similarity in specificity
sub_metric: sub_metric in metric to use compute the similarity in specificity
'''
def get_coverability(references, predictions, metric='bertscore', sub_metric='f1'):

    num_papers = len(references)
    coverability = 0

    # loop through all papers
    for ref, pre in zip(references, predictions):

        ref_array = np.array(ref)
        pre_array = np.array(pre)
        assert len(ref_array) == len(pre_array) and len(ref_array) > 1

        # get all distinct combinations
        idx = np.stack(np.triu_indices(len(ref_array), k=1), axis=-1)

        ref_array_ref = ref_array[idx[:, 0]]
        ref_array_pre = ref_array[idx[:, 1]]

        pre_array_ref = pre_array[idx[:, 0]]
        pre_array_pre = pre_array[idx[:, 1]]

        h, g = 0, 0

        # compute the pairwise similarities
        for r, p in zip(ref_array_ref, ref_array_pre):
            h += get_results([r], [p], [metric])[metric][sub_metric]
        for r, p in zip(pre_array_ref, pre_array_pre):
            g += get_results([r], [p], [metric])[metric][sub_metric]

        h /= len(ref_array) * (len(ref_array) - 1) / 2
        g /= len(pre_array) * (len(pre_array) - 1) / 2

        coverability += g - h

    coverability /= num_papers
        
    return coverability


if __name__ == "__main__":

    references = [['you are nice', 'you are good'], ['you are bad', 'you are not good']]
    predictions = ['you are nice 1', 'you are nice 2']

    print("================Main Results================")
    print(get_results(references, predictions))
    print("================Specificity================")
    print(get_specificity(references, predictions))

    references = [['you are nice', 'you are good'], ['you are bad', 'you are not good']]
    predictions = [['you are nice 1', 'you are nice 2'], ['you are nice 1', 'you are nice 2']]

    print("================Coverability================")
    print(get_coverability(references, predictions))
