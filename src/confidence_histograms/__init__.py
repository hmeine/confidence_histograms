# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# **InsertLicense** code author="Hans Meine"

from typing import List, Optional, Tuple, Union
import numpy as np


class ConfidenceHistograms:
    '''Collects 4D histogram based on classifier output and reference labels:
    1. dimension is the image / case (allowing to compute statistics over cases)
    2. dimension is the output label
    3. dimension is a binary condition whether the label was correct or not
    4. dimension contains the fine-binned classifier confidence in that label

    Therefore:
    * histograms[:,2,1,:] are the confidences for true positives of label 2.
    * histograms[:,2,0,:] are the confidences for false positives of label 2.
    * histograms[:,label!=2,1,:].sum(1) is the same if all labels add up to 1.0.

    If we would assume that the classifier output always adds up to 1.0 over all
    labels, we could work with less histograms, but this way we need to make
    less assumptions.

    The class constant `MIN_SAMPLES` affects all reliability diagrams and
    specifies the minimum number of samples that need to be in a (re-binned)
    accumulator bin in order to compute a valid observed frequency.  (If the
    number of samples is too low, the plots will look strange because of bad
    estimations of the correct/total quotient.)

    The two class attributes ADAPTIVE_CONFIDENCE and SMOOTH_HISTOGRAMS can be
    used to toggle the behavior of returning confidence values averaged over the
    histogram entries (on by default) instead of a fixed-distance array, or
    whether to use channel encoding to partially distribute entries close to bin
    edges among two adjacent bins (off by default).  These two options are
    exclusive, since the former does not make sense anymore after channel
    encoding.
    '''

    MIN_SAMPLES = 50
    ADAPTIVE_CONFIDENCE = True
    SMOOTH_HISTOGRAMS = False

    def __init__(self, label_histograms: Union[List[np.ndarray], np.ndarray],
                 prediction_histograms: Union[List[np.ndarray], np.ndarray],
                 uncertainty_histograms: Union[List[np.ndarray], np.ndarray]):
        self._label_histograms = label_histograms
        self._prediction_histograms = prediction_histograms
        self._uncertainty_histograms = uncertainty_histograms

        if not len(prediction_histograms) and len(label_histograms):
            assert self.label_count() == 2
        elif len(prediction_histograms) != len(label_histograms):
            raise ValueError('confidence histograms must have the same number of cases')
        if len(uncertainty_histograms) != len(label_histograms):
            raise ValueError('uncertainty and confidence histograms must have the same number of cases')

    @property
    def label_histograms(self):
        return np.asarray(self._label_histograms)

    @property
    def prediction_histograms(self):
        if not len(self._prediction_histograms):
            assert self.label_count() == 2
            # for binary classification, we know that every confidence >= 0.5 is a prediction:
            result = self.label_histograms.sum(1)
            result[:,:,:result.shape[-1]//2] = 0
            return result
        return np.asarray(self._prediction_histograms)

    @property
    def uncertainty_histograms(self):
        return np.asarray(self._uncertainty_histograms)

    def add_case_predictions(self, predictions: np.ndarray, reference: np.ndarray,
                             weights: Optional[np.ndarray] = None, internal_bin_count: int = 2**14):
        assert np.ndim(predictions) >= 2, 'expecting categorical vectors for both inputs'
        assert predictions.shape == reference.shape
        label_count = predictions.shape[-1]
        assert label_count >= 2, 'expecting categorical vectors with >=2 classes'
        assert internal_bin_count >= 10, 'internal_bin_count should be a large positive integer'

        if weights is not None:
            assert weights.shape == reference.shape[:-1], 'weights will be broadcasted over labels'

        predictions = predictions.reshape((-1, predictions.shape[-1]))
        reference = reference.reshape((-1, reference.shape[-1]))
        weights = weights.ravel() if weights is not None else None

        predicted_label = np.argmax(predictions, axis = -1)
        reference_label = np.argmax(reference, axis = -1)
        correctness = (predicted_label == reference_label).astype(np.uint8)
        histogram2d_range = np.array([[0.0, 1.0], [0.0, 1.0]])

        case_label_hist = []
        for label in range(label_count):
            case_label_hist.append(
                np.histogram2d(reference_label == label, predictions[:,label],
                                  bins = [2, internal_bin_count],
                                  range = histogram2d_range, weights = weights)[0]
            )

        if label_count > 2:
            pred_confidence = np.choose(predicted_label, predictions.T)
            case_prediction_hist = (
                np.histogram2d(correctness, pred_confidence,
                                  bins = [2, internal_bin_count],
                                  range = histogram2d_range, weights = weights)[0]
            )
        else:
            # optimization: for binary problems, prediction_histogram() can
            # be computed based on self._label_histograms alone
            case_prediction_hist = np.array([])

        normalized_entropy = -((predictions * np.log(predictions)).sum(-1)
                               / np.log(label_count))
        case_uncertainty_hist = np.histogram2d(
            correctness, normalized_entropy,
            bins = [2, internal_bin_count],
            range = histogram2d_range, weights = weights)[0]

        self.add_case_accumulators(
            np.asanyarray(case_label_hist), case_prediction_hist, case_uncertainty_hist)

    def add_case_accumulators(self,
                              case_label_hist: np.ndarray,
                              case_prediction_hist: np.ndarray,
                              case_uncertainty_hist: np.ndarray):
        '''Add 3D accumulator arrays with shape (labelCount, 2,
        internalBinCount) for the per-label histograms and (2, internalBinCount)
        for the prediction histogram. The first (label) dimension depends on the
        channel extent of the first input. The second dimension separates the
        accumulators for the wrong/true labels. The third dimension spans the
        confidence range from 0..1. Increases the number of cases by one.
        '''
        assert np.ndim(case_label_hist) == 3
        assert np.shape(case_label_hist)[1] == 2
        # FIXME: what about the optimization for label_count == 2?
        assert np.shape(case_label_hist)[1:] == np.shape(case_prediction_hist)
        assert len(self._label_histograms) == len(self._prediction_histograms)

        if not isinstance(self._label_histograms, list):
            self._label_histograms = list(self._label_histograms)
        if not isinstance(self._prediction_histograms, list):
            self._prediction_histograms = list(self._prediction_histograms)
        if not isinstance(self._uncertainty_histograms, list):
            self._uncertainty_histograms = list(self._uncertainty_histograms)

        if self.case_count():
            # shape needs to be compatible with existing entries:
            assert np.shape(case_label_hist) == np.shape(self._label_histograms[0])
            assert np.shape(case_prediction_hist) == np.shape(self._prediction_histograms[0])
            assert np.shape(case_uncertainty_hist) == np.shape(self._uncertainty_histograms[0])

        self._label_histograms.append(np.asarray(case_label_hist))
        self._prediction_histograms.append(np.asarray(case_prediction_hist))
        self._uncertainty_histograms.append(np.asarray(case_uncertainty_hist))

    def __len__(self) -> int:
        return self.case_count()

    def __getitem__(self, index: Union[int, slice]) -> 'ConfidenceHistograms':
        '''same as case(index)'''
        return self.case(index)

    def case_count(self) -> int:
        return len(self._label_histograms)

    def label_count(self) -> int:
        return len(self._label_histograms[0])

    def internal_bin_count(self) -> int:
        return self._label_histograms[0].shape[-1]

    def case(self, index: Union[int, slice]) -> 'ConfidenceHistograms':
        '''Return a new ConfidenceHistograms instance which only contains the data of
        a certain case (or slice of cases).'''
        if isinstance(index, int):
            index = slice(index, index + 1)
        return ConfidenceHistograms(
            self.label_histograms[index],
            self.prediction_histograms[index],
            self.uncertainty_histograms[index])

    def label(self, label: Union[int, slice]) -> 'ConfidenceHistograms':
        '''Return a new ConfidenceHistograms instance which only looks at
        a certain label (or slice of labels).'''
        if isinstance(label, int):
            label = slice(label, label + 1)
        return ConfidenceHistograms(
            self.label_histograms[:,label],
            self.prediction_histograms[:,label],
            self.uncertainty_histograms)

    def save(self, filename: str) -> None:
        np.savez_compressed(
            filename,
            self.label_histograms,
            self.prediction_histograms,
            self.uncertainty_histograms
        )

    def load(self, filename: str) -> None:
        with np.load(filename) as state:
            histograms = tuple(state.values())
            self._label_histograms, self._prediction_histograms = histograms[:2]
            self._uncertainty_histograms = histograms[2] if len(histograms) > 2 else []
        if np.ndim(self._label_histograms) != 4:
            raise ValueError(f'loaded confidence histograms per label have shape {self._label_histograms.shape}, expected 4D')
        if np.ndim(self._prediction_histograms) != 3:
            raise ValueError(f'loaded confidence histograms for the prediction have shape {self._prediction_histograms.shape}, expected 3D')
        if len(self._uncertainty_histograms) and np.ndim(self._uncertainty_histograms) != 3:
            raise ValueError(f'loaded uncertainty histograms have shape {np.shape(self._uncertainty_histograms)}, expected 3D')

    @classmethod
    def create_empty(cls) -> 'ConfidenceHistograms':
        return cls([], [], [])

    @classmethod
    def from_file(cls, filename: str) -> 'ConfidenceHistograms':
        result = cls.create_empty()
        result.load(filename)
        return result

    @staticmethod
    def rebin(array: np.ndarray, bins = 16) -> np.ndarray:
        '''Rebin confidence (last axis) down to more coarse sampling.

        Optionally, perform channel smoothing (each confidence entry between two
        bin centers is linearly distributed among these two bins).  This
        behavior can be globally switched off via the boolean class attribute
        ConfidenceHistograms.SMOOTH_HISTOGRAMS, but requires switching off
        ConfidenceHistograms.ADAPTIVE_CONFIDENCE.
        '''
        new_shape = array.shape[:-1] + (bins, -1)
        reshaped = array.reshape(new_shape)
        if ConfidenceHistograms.SMOOTH_HISTOGRAMS:
            # the right half of each bin (except the last) shall be moved right, with a lambda ranging from 0 to 0.5:
            to_right = reshaped[...,:-1,:] * np.linspace(-0.5, 0.5, reshaped.shape[-1]).clip(0, 1)
            # the left half of each bin (except the first) shall be moved right, with a lambda ranging from 0.5 to 0:
            to_left  = reshaped[...,1: ,:] * np.linspace(0.5, -0.5, reshaped.shape[-1]).clip(0, 1)
            reshaped[...,:-1,:] += to_left - to_right
            reshaped[...,1:,:] += to_right - to_left
        return reshaped.sum(-1)

    def calibration_errors(self, bins: int = 16, label: Union[int, str] = 'predicted') -> Tuple[float, float, float]:
        '''Return triple of (expected_calibration_error(),
        unweighted_expected_calibration_error(), maximum_calibration_error())
        but in a more efficient way than using these three frontend functions
        individually
        '''
        confidence, reliability, bin_totals = self.reliability_diagram(bins, label = label)
        calibrationError = np.abs(reliability - confidence)
        ece = (bin_totals * calibrationError).sum() / bin_totals.sum()
        uece = calibrationError.mean()
        return ece, uece, calibrationError.max()

    def expected_calibration_error(self, bins: int = 16, label: Union[int, str] = 'predicted') -> float:
        '''The expected calibration error (ECE) is the weighted average distance
        of the reliability diagram from the diagonal
        '''
        result, _, _ = self.calibration_errors(bins, label)
        return result

    def unweighted_expected_calibration_error(self, bins: int = 16, label: Union[int, str] = 'predicted') -> float:
        '''The unweighted expected calibration error (uECE) is the average
        distance of the reliability diagram from the diagonal, not taking into
        account the number of samples per bin
        '''
        _, result, _ = self.calibration_errors(bins, label)
        return result

    def uncertainty_calibration_error(self, bins: int = 16) -> float:
        '''The uncertainty calibration error (UCE) according to Laves et al.
        (2019) is based on the normalized entropy as an uncertainty measure
        computed over all labels, compared against the classification error
        '''
        return self.expected_calibration_error(bins, label = 'uncertainty')

    def maximum_calibration_error(self, bins: int = 16, label: Union[int, str] = 'predicted') -> float:
        '''The maximum calibration error (MCE) is the maximum
        distance of the reliability diagram from the diagonal
        '''
        _, _, result = self.calibration_errors(bins, label)
        return result

    def _internal_bin_confidences(self) -> np.ndarray:
        internal_bins = self.internal_bin_count()
        return (np.arange(internal_bins) + 0.5) / internal_bins

    def reliability_diagram(self, bins: int = 16, label: Union[int, str] = 'predicted',
            adaptive_binning: bool = False) -> Tuple[np.ndarray, np.ma.MaskedArray, np.ma.MaskedArray]:
        '''
        Return (confidence, observed_frequency, bin_totals) tuple of ndarrays
        for plotting reliability diagrams.  If label == 'predicted', the diagram
        is based on the confidence in the predicted class only.  If label is
        specified as an integer, the diagram is for the given label only. If
        label == 'all', the diagram is aggregated over all labels.  If label ==
        'uncertainty', the x-axis is the normalized entropy over all labels (cf.
        `uncertainty_histogram`) and the y-axis is inverted accordingly [Laves+
        2019] `bin_totals` is the sum in the respective bin (sample count or sum
        of weights) and needed for weighted ECE computation.
        '''
        assert self.MIN_SAMPLES > 0
        if label == 'predicted':
            confidence_histograms = self.prediction_histograms
        elif label == 'all':
            confidence_histograms = self.label_histograms
        elif label == 'uncertainty':
            confidence_histograms = self.uncertainty_histograms
        elif isinstance(label, int):
            confidence_histograms = self.label_histograms[:,label]
        else:
            raise ValueError("label must be 'predicted', 'all', or a valid integer label")

        while np.ndim(confidence_histograms) > 2:
            confidence_histograms = confidence_histograms.sum(0)

        if adaptive_binning:
            rebin = self.adaptive_binning
        else:
            rebin = self.rebin

        if self.ADAPTIVE_CONFIDENCE:
            binned = rebin(np.vstack((
                confidence_histograms,
                self._internal_bin_confidences() * confidence_histograms.sum(0)
            )), bins)
            confidence = binned[2] / (binned[:2].sum(0)).clip(1e-8, None)
        else:
            binned = rebin(confidence_histograms, bins)
            confidence = (np.arange(bins) + 0.5) / bins

        correct_classification = binned[1]
        bin_totals = binned[:2].sum(0)
        observed_frequency = correct_classification / bin_totals.clip(1, None)

        if label == 'uncertainty':
            observed_frequency = 1 - observed_frequency

        bad_mask = bin_totals < self.MIN_SAMPLES
        return np.ma.masked_array(confidence, bad_mask), np.ma.masked_array(observed_frequency, bad_mask), bin_totals
    
    def adaptive_binning(self, histogram: np.ndarray, bins: int = 16) -> np.ndarray:
        histogram_1d = histogram
        while np.ndim(histogram_1d) > 1:
            histogram_1d = histogram_1d.sum(0)

        bin_indices = self.adaptive_bin_edges(histogram_1d, bins)

        cumulative_entries = np.cumsum(histogram, -1)

        right_bin_indices = bin_indices[1:]
        binned = cumulative_entries[...,right_bin_indices-1]
        left_bin_indices = bin_indices[:-1]
        assert left_bin_indices[0] == 0, 'cumulative_entries[0] will already have the sum over one entry'
        binned[...,1:] -= cumulative_entries[...,left_bin_indices[1:]]

        return binned

    def adaptive_bin_edges(self, histogram: np.ndarray, bins: int = 16) -> np.ndarray:
        '''
        Given a `histogram`, iteratively chunks of a number of bins to the left
        and to the right, trying to take at least 1/nth of the remaining amount,
        with n being the remaining number of bins.
        
        Returns an ndarray with (bins+1) indices, starting with 0, ending with
        len(array).
        '''
        assert np.ndim(histogram) == 1
        cumulative_entries = np.cumsum(histogram)

        left_bin_edges = [0]
        right_bin_edges = [len(cumulative_entries)]
        for i in range(bins - 1):
            left_index = left_bin_edges[-1]
            right_index = right_bin_edges[-1]
            left_value = cumulative_entries[left_index-1] if left_index > 0 else 0
            right_value = cumulative_entries[right_index-1]
            remaining_amount = right_value - left_value
            chunk_amount = remaining_amount / (bins - i)

            if i % 2 == 0:
                next_left_index = left_index + 1 + np.searchsorted(
                    cumulative_entries[left_index+1:right_index], left_value + chunk_amount,
                )
                left_bin_edges.append(next_left_index)
            else:
                next_right_index = left_index + np.searchsorted(
                    cumulative_entries[left_index:right_index], right_value - chunk_amount,
                )
                right_bin_edges.append(next_right_index)
                
        return np.asarray(left_bin_edges + right_bin_edges[::-1])

    def reliability_per_case(self, bins: int = 16, label: Union[int, str] = 'predicted') -> Tuple[np.ndarray, np.ma.MaskedArray]:
        assert np.ndim(self.label_histograms) == 4, 'reliability_per_case() requires per-case info'

        result_confidence = []
        result_reliability = []
        for case_index in range(self.case_count()):
            case_confidence, case_reliability, _ = self.case(case_index).reliability_diagram(bins, label)
            result_confidence.append(case_confidence)
            result_reliability.append(case_reliability)

        return np.ma.asarray(result_confidence).mean(0), np.ma.asarray(result_reliability)

    @staticmethod
    def _setup_unit_square_axes(mpl_ax, margin = 0.0):
        mpl_ax.plot([0, 1], [0, 1], c = 'k', ls = ':')

        mpl_ax.set_aspect(1.0)
        mpl_ax.set_xlim(0 - margin, 1 + margin)
        mpl_ax.set_ylim(0 - margin, 1 + margin)
        if False:
            # the positions of the boxplots are odd and long:
            for l in mpl_ax.get_xticklabels(): l.update(dict(rotation = 90))
        else:
            # set up cleaner tick positions:
            xticks = (0, 0.25, 0.5, 0.75, 1)
            mpl_ax.set_xticks(xticks)
            mpl_ax.set_xticklabels(xticks)
            # for symmetry, reuse ticks for y:
            mpl_ax.set_yticks(xticks)
            mpl_ax.set_yticklabels(xticks)

    @classmethod
    def setup_reliability_diagram_axes(cls, mpl_ax, label: Union[int, str]):
        cls._setup_unit_square_axes(mpl_ax)
        if label != 'uncertainty':
            multiclass = label != 'predicted'
            mpl_ax.set_xlabel('confidence')
            mpl_ax.set_ylabel('observed frequency' if multiclass else 'accuracy')
        else:
            mpl_ax.set_xlabel('uncertainty')
            mpl_ax.set_ylabel('error')

    def plot_reliability_diagram(self, mpl_ax = None, bins: int = 16, per_case_boxplots: bool = True,
                                 by_label: bool = False, label: Union[int, str] = 'predicted',
                                 adaptive_binning: bool = False,
                                 summary_statistics = True, temperature_label: Optional[str] = None):
        if mpl_ax is None:
            import matplotlib.pyplot as plt
            mpl_ax = plt.gca()

        confidence, total_reliability, _ = self.reliability_diagram(bins, label, adaptive_binning = adaptive_binning)

        if per_case_boxplots and self.case_count() > 1:
            _, reliability = self.reliability_per_case(bins, label = label)
            # boxplot() does not support masked arrays, so we need to remove masked values manually:
            mpl_ax.boxplot([r.compressed() for r in reliability.T], positions = confidence, widths= 0.68/bins,
                           # de-emphasize outlier markers:
                           flierprops = dict(alpha=0.25, markersize=3), sym = '.')
        
        if by_label:
            assert label == 'predicted', 'label and by_label should not be specified at the same time'
            for each_label in range(self.label_count()):
                confidence, total_reliability, _ = self.reliability_diagram(bins, label = each_label)
                mpl_ax.plot(confidence, total_reliability, lw = 2, label = f'label {each_label}')
            mpl_ax.legend()
        else:
            mpl_ax.plot(confidence, total_reliability, lw = 2)

        if summary_statistics:
            nll = self.negative_log_likelihood()
            ece, uece, mce = self.calibration_errors(bins, label = 'all')
            uce = self.uncertainty_calibration_error(bins)
            labels = [
                f'NLL = {nll:.3f}',
                f'ECE = {ece:.3f}',
                f'uECE = {uece:.3f}',
                f'UCE = {uce:.3f}'
            ]
            if temperature_label:
                labels.append(temperature_label)
            mpl_ax.text(0.05, 0.95, '\n'.join(labels),
                        verticalalignment = 'top', horizontalalignment = 'left',
                        transform = mpl_ax.transAxes, fontsize = 12)

        self.setup_reliability_diagram_axes(mpl_ax, label)

    def roc_curve(self, label: Union[int, str]) -> Tuple[np.ndarray, np.ndarray]:
        '''Return (fpr, tpr) for the given label.
        
        label = "micro" means micro-averaging all ROC curves for a multiclass problem.
        (macro-averaging is currently not implemented because it would require
        interpolation, for which scipy would be convenient, but we do not want to
        depend on that.)'''
        
        assert np.ndim(self.label_histograms) == 4, ('while ConfidenceHistograms '
            'in general supports selecting cases or labels, this method does not yet')

        if label == 'macro':
            label_curves = [self.roc_curve(label) for label in range(self.label_count())]
            raise NotImplementedError('macro averaging would require interpolating '
                                      'all tpr to a common fpr grid')
        
        histograms = self.label_histograms.sum(0)

        if label == 'micro':
            labelHist = histograms.sum(0)
        else:
            labelHist = histograms[label]
        
        truePositives  = labelHist[1,::-1].cumsum(-1)
        falsePositives = labelHist[0,::-1].cumsum(-1)

        totalPositiveCount = truePositives[-1]
        totalNegativeCount = falsePositives[-1]

        tpr = truePositives.astype(float)  / max(totalPositiveCount, 1)
        fpr = falsePositives.astype(float) / max(totalNegativeCount, 1)

        return fpr, tpr
    
    @classmethod
    def setup_roc_curve_axes(cls, mpl_ax):
        cls._setup_unit_square_axes(mpl_ax, margin = 0.02)
        mpl_ax.set_xlabel('false positive rate (1 - specificity)')
        mpl_ax.set_ylabel('true positive rate (sensitivity')

    def plot_roc_curve(self, label: Union[int, str], mpl_ax = None):
        if mpl_ax is None:
            import matplotlib.pyplot as plt
            mpl_ax = plt.gca()

        fpr, tpr = self.roc_curve(label)

        mpl_ax.plot(fpr, tpr)
        self.setup_roc_curve_axes(mpl_ax)

    def negative_log_likelihood(self) -> float:
        '''Computes NLL (which is effectively the same as categorical cross
        entropy) based on the internal histogram representation'''
        histograms = self.label_histograms
        log_confidence = np.log(self._internal_bin_confidences())
        return -(histograms[...,1,:] * log_confidence).sum() / histograms[...,1,:].sum()

    def multiclass_brier_score(self) -> float:
        # https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes

        histograms = self.label_histograms
        # sum over cases and labels:
        histograms_2d = histograms.reshape((-1, ) + histograms.shape[-2:]).sum(0)
        confidence = self._internal_bin_confidences()

        false_prob, true_prob = histograms_2d
        total_count = max(histograms_2d.sum(), 1e-13) / self.label_count() # prevent DIV0

        # we have binned confidence for true label = 0/1 separately, so
        # we can subtract confidence from 0 / 1 and use the histogram as weights:
        return (np.sum((1 - confidence) ** 2 * true_prob) +
                np.sum((    confidence) ** 2 * false_prob)) / total_count
