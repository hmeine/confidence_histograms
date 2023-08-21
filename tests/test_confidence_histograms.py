# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# **InsertLicense** code author="Hans Meine"

import numpy as np
from .confidence_histograms import ConfidenceHistograms


# test data for an "interesting" ROC curve, based on iris dataset / random noise
# see https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
Y_SCORES = np.array(
      [[0.20904258, 0.44281479, 0.72972079],
       [0.53826472, 0.28628506, 0.55947165],
       [0.72628705, 0.18578229, 0.46880349],
       [0.1240331 , 0.58422834, 0.66991616],
       [0.65044826, 0.49312323, 0.23494465],
       [0.30188469, 0.45310089, 0.62260347],
       [0.67040757, 0.51312554, 0.19254017],
       [0.29842315, 0.76423344, 0.32118777],
       [0.43641426, 0.51574956, 0.42842427],
       [0.38096532, 0.5058534 , 0.49548708],
       [0.40910573, 0.47928401, 0.49264344],
       [0.45998067, 0.24423577, 0.67185585],
       [0.5235907 , 0.28089618, 0.57799679],
       [0.61087502, 0.38486006, 0.38741775],
       [0.34359126, 0.50123477, 0.53944279],
       [0.72841338, 0.3240822 , 0.32246985],
       [0.23806955, 0.63218646, 0.51435456],
       [0.3334454 , 0.57607106, 0.47282672],
       [0.88004588, 0.07182615, 0.42750095],
       [0.59178918, 0.12069485, 0.67537248],
       [0.24184594, 0.6192129 , 0.53122374],
       [0.64295412, 0.38748743, 0.35216679],
       [0.50995947, 0.18682115, 0.68119745],
       [0.73141863, 0.24024936, 0.41586544],
       [0.2565427 , 0.53526745, 0.58624955],
       [0.61002336, 0.44154852, 0.31821984],
       [0.63786598, 0.5889047 , 0.14920642],
       [0.62792013, 0.22483454, 0.5297054 ],
       [0.58082697, 0.43580659, 0.36346956],
       [0.54369648, 0.23970523, 0.59208052],
       [0.44448723, 0.41389564, 0.52338278],
       [0.52303467, 0.19336243, 0.66901752],
       [0.62062251, 0.37744495, 0.39058105],
       [0.34904125, 0.38395961, 0.65479605],
       [0.42200901, 0.23740453, 0.71633628],
       [0.45118652, 0.4742681 , 0.45727186],
       [0.62312526, 0.3553002 , 0.41173434],
       [0.        , 0.65264553, 0.7331334 ],
       [0.7720679 , 0.27084965, 0.33638753],
       [0.6317612 , 0.30445353, 0.44622359],
       [0.37456096, 0.52831727, 0.4804915 ],
       [0.75926624, 0.06359664, 0.55058425],
       [0.41384814, 0.30448146, 0.66202918],
       [0.65081625, 0.21199813, 0.52242508],
       [0.66666112, 0.17821262, 0.53727825],
       [0.58694833, 0.15921986, 0.62276467],
       [0.24123838, 0.48889742, 0.65003792],
       [0.20315833, 0.4032115 , 0.7822074 ],
       [0.296     , 0.56858441, 0.5201967 ],
       [0.09142642, 0.28827815, 1.        ],
       [0.47478098, 0.48338165, 0.42566162],
       [0.46940947, 0.45364017, 0.46862585],
       [0.44971977, 0.26413176, 0.66150742],
       [0.72820065, 0.29016522, 0.36022322],
       [0.26698176, 0.5897394 , 0.5319579 ],
       [0.33322074, 0.45236797, 0.59505141],
       [0.33001892, 0.57842457, 0.477274  ],
       [0.41563883, 0.34884924, 0.61323171],
       [0.36660546, 0.52002215, 0.49377602],
       [0.69756984, 0.15533882, 0.52982007],
       [0.52175395, 0.37073477, 0.49075031],
       [0.83513885, 0.16741213, 0.37663598],
       [0.76058134, 0.1779454 , 0.43652131],
       [0.54509228, 0.24118497, 0.59689038],
       [0.40507722, 0.42278425, 0.55756273],
       [0.25918955, 0.71098238, 0.42241129],
       [0.4635024 , 0.32486759, 0.58610752],
       [0.0868191 , 0.38409127, 0.90051328],
       [0.66926084, 0.3228693 , 0.39505862],
       [0.60899295, 0.32752787, 0.45017226],
       [0.41355163, 0.28298859, 0.68540896],
       [0.47831232, 0.60753653, 0.29379792],
       [0.76778424, 0.13568688, 0.48255789],
       [0.80357607, 0.18434717, 0.39677547],
       [0.42227869, 0.48664587, 0.4784463 ]])

Y_TEST = np.array(
    [2, 1, 0, 2, 0, 2, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 1, 1, 0, 0, 2, 1,
     0, 0, 2, 0, 0, 1, 1, 0, 2, 1, 0, 2, 2, 1, 0, 1, 1, 1, 2, 0, 2, 0,
     0, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 0, 2, 1, 1, 1,
     1, 2, 0, 0, 2, 1, 0, 0, 1])[:,np.newaxis] == np.arange(3)


def test_reliability_diagram_adding():
    ch = ConfidenceHistograms.create_empty()
    ch.MIN_SAMPLES = 1
    assert ch.case_count() == 0
    ch.add_case_predictions(Y_SCORES, Y_TEST)
    assert ch.case_count() == 1
    assert ch.label_count() == 3
    assert ch.prediction_histograms.sum() == len(Y_SCORES)
    assert ch.label_histograms.sum() == len(Y_SCORES)*3
    ece1, uece1, mce1 = ch.calibration_errors()

    ch.add_case_predictions(Y_SCORES, Y_TEST)
    assert ch.case_count() == 2
    assert ch.prediction_histograms.sum() == len(Y_SCORES)*2
    assert ch.label_histograms.sum() == len(Y_SCORES)*2*3
    assert ch.prediction_histograms[1].sum() == len(Y_SCORES)
    assert ch.label_histograms[1].sum() == len(Y_SCORES)*3
    ece2, uece2, mce2 = ch.calibration_errors()
    assert abs(ece1 - ece2) < 1e-10
    assert abs(uece1 - uece2) < 1e-10
    assert abs(mce1 - mce2) < 1e-10


def test_reliability_diagram_standard():
    ch = ConfidenceHistograms.create_empty()
    ch.add_case_predictions(Y_SCORES, Y_TEST)

    ch.MIN_SAMPLES = 1

    ece_sum = 0.0
    mce = 0.0

    bin_confidence, accuracy, bin_totals = ch.reliability_diagram()
    predicted_confidence = Y_SCORES.max(axis = 1)
    correctness = np.argmax(Y_SCORES, axis = 1) == np.argmax(Y_TEST, axis = 1)
    for bin_index in range(16):
        samples_in_bin = (predicted_confidence >= bin_index / 16)
        if bin_index < 15:
            samples_in_bin *= (predicted_confidence < (bin_index+1) / 16)

        assert (samples_in_bin.sum() > 0) == (bin_index >= 7)
        assert bin_totals[bin_index] == samples_in_bin.sum()

        if not samples_in_bin.any():
            assert np.ma.is_masked(accuracy[bin_index])
        else:
            bin_accuracy = correctness[samples_in_bin].mean()
            assert abs(bin_accuracy - accuracy[bin_index]) < 1e-10

            bin_error = abs(bin_confidence[bin_index] - bin_accuracy)
            mce = max(mce, bin_error)
            ece_sum += bin_error * samples_in_bin.sum()

    ece = ece_sum / len(Y_SCORES)
    assert abs(ch.expected_calibration_error(16) - ece) < 1e-10
    assert abs(ch.maximum_calibration_error(16) - mce) < 1e-10


def test_binary_case_optimization():
    ch = ConfidenceHistograms.create_empty()
    ch.add_case_predictions(Y_SCORES[:15,:2], Y_TEST[:15,:2])
    ch.add_case_predictions(Y_SCORES[15:,:2], Y_TEST[15:,:2])

    bin_confidence, accuracy, bin_totals = ch.reliability_diagram()
    ece = ch.expected_calibration_error(16)
    mce = ch.maximum_calibration_error(16)


def test_nll():
    ch = ConfidenceHistograms.create_empty()
    ch.add_case_predictions(Y_SCORES, Y_TEST, internal_bin_count = 2**14)

    try:
        import scipy.special
        # compute NLL and expected maximal error range taking quantization into account
        # bin size is 1/2^14, maximal quantization error is thus 1/2^15
        nll_scipy, nll_scipy_min, nll_scipy_max = (
            np.mean(
                -scipy.special.xlogy(Y_TEST, (Y_SCORES - ofs).clip(0, 1)).sum(-1)
            )
            for ofs in (0, -1/2**15, 1/2**15)
        )
        #print(nll_scipy, nll_scipy_min, nll_scipy_max)
    except ImportError:
        nll_scipy     = 0.6511194748126142
        nll_scipy_min = 0.6510563492355167
        nll_scipy_max = 0.651183012220931

    print(f'NLL should be {nll_scipy:.6g} or at least in range {nll_scipy_min:.6g}..{nll_scipy_max:.6g}')
    assert ch.negative_log_likelihood() >= nll_scipy_min
    assert ch.negative_log_likelihood() <= nll_scipy_max


BRIER_SCORES = np.array(
    [[0.14, 0.38, 0.4 , 0.04, 0.05],
     [0.55, 0.05, 0.34, 0.04, 0.01],
     [0.3 , 0.35, 0.18, 0.09, 0.08],
     [0.23, 0.22, 0.04, 0.05, 0.46],
     [0.  , 0.15, 0.47, 0.28, 0.09],
     [0.23, 0.13, 0.34, 0.27, 0.03],
     [0.32, 0.06, 0.59, 0.02, 0.01],
     [0.01, 0.19, 0.01, 0.03, 0.75],
     [0.27, 0.38, 0.03, 0.12, 0.2 ],
     [0.17, 0.45, 0.11, 0.25, 0.01]]
)

BRIER_TEST = np.array(
    [[0, 0, 0, 0, 1],
     [0, 0, 0, 0, 1],
     [0, 0, 0, 0, 1],
     [0, 1, 0, 0, 0],
     [0, 0, 0, 0, 1],
     [0, 0, 1, 0, 0],
     [1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [1, 0, 0, 0, 0],
     [1, 0, 0, 0, 0]]
)

def test_brier_score():
    ch = ConfidenceHistograms.create_empty()
    ch.add_case_predictions(BRIER_SCORES, BRIER_TEST, internal_bin_count = 2**14)

    # https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
    assert np.isclose(ch.multiclass_brier_score(), 1.00689, 1.5e-5)
