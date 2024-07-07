import numpy as np
import matplotlib.pyplot as plt


def get_filter(name, T, rolloff=None):
    def rc(t, beta):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.sinc(t)*np.cos(np.pi*beta*t)/(1-(2*beta*t)**2)

    def rrc(t, beta):
        return (np.sin(np.pi*t*(1-beta))+4*beta*t*np.cos(np.pi*t*(1+beta)))/(np.pi*t*(1-(4*beta*t)**2))

    # rolloff is ignored for triang and rect
    if name == 'rect':
        return lambda t: (abs(t/T) < 0.5).astype(int)
    if name == 'triang':
        return lambda t: (1-abs(t/T)) * (abs(t/T) < 1).astype(float)
    elif name == 'rc':
        return lambda t: rc(t/T, rolloff)
    elif name == 'rrc':
        return lambda t: rrc(t/T, rolloff)


T = 1
Fs = 100


def get_signal(g, d):
    """Generate the transmit signal as sum(d[k]*g(t-kT))"""
    t = np.arange(-2*T, (len(d)+2)*T, 1/Fs)
    g0 = g(np.array([1e-8]))
    xt = sum(d[k]*g(t-k*T) for k in range(len(d)))
    return t, xt/g0


def drawFullEyeDiagram(xt):
    """Draw the eye diagram using all parts of the given signal xt"""
    samples_perT = Fs*T
    samples_perWindow = 2*Fs*T
    parts = []
    startInd = 2*samples_perT   # ignore some transient effects at beginning of signal

    for k in range(int(len(xt)/samples_perT) - 6):
        parts.append(xt[startInd + k*samples_perT +
                     np.arange(samples_perWindow)])
    parts = np.array(parts).T

    t_part = np.arange(-T, T, 1/Fs)
    plt.plot(t_part, parts, 'b-')


def drawSignals(g, data=None):
    """Draw the transmit signal, the used filter and the resulting eye-diagram
    into one figure."""
    N = 100
    if data is None:
        data = 2*((np.random.randn(N) > 0))-1
        # fix the first 10 elements for  keeping the shown graphs constant
        # between eye diagrams
        data[0:10] = 2*np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])-1

    t, xt = get_signal(g, data)

    plt.subplot(223)
    t_g = np.arange(-4*T, 4*T, 1/Fs)
    plt.plot(t_g, g(t_g))

    plt.subplot(211)
    plt.plot(t, xt)
    plt.stem(data)

    plt.subplot(224)
    drawFullEyeDiagram(xt)
    plt.ylim((-2, 2))
    plt.tight_layout()
    plt.show()


def showRCEyeDiagram(alpha):
    g = get_filter('rc', T=1, rolloff=alpha)
    drawSignals(g)


showRCEyeDiagram(alpha=1)
