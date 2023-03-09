import torch
import essentia
import essentia.standard as es

class FeatureExtractor:

    def __init__(self):
        super().__init__()

    def extract(self, audio, sample_rate = 44100):
        pool = essentia.Pool()
        audio = audio.numpy()

        spectrum = es.Spectrum()
        w = es.Windowing(type = 'hann')
        
        for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512, startFromZero=True):
            spec = spectrum(w(frame))
            freqs, mags = es.SpectralPeaks(minFrequency=1)(spec)
            pool.add('decrease', es.Decrease()(spec))
            # pool.add('rolloff', es.RollOff()(spec))
            pool.add('centroid', es.Centroid()(spec))
            pool.add('odd_even', es.OddToEvenHarmonicEnergyRatio()(freqs, mags))
            # pool.add('flux', es.Flux()(spec))
        aggrPool = es.PoolAggregator(defaultStats = [ 'median', 'stdev' ])(pool)
        
        lat, _, _ = es.LogAttackTime()(audio)
        centroid = aggrPool['centroid.median']
        decrease = aggrPool['decrease.median']
        # rolloff = aggrPool['rolloff.median']
        odd_even = aggrPool['odd_even.median']
        # flux = aggrPool['flux.median']

        features = torch.tensor([lat, centroid, rolloff, odd_even])
        return features 