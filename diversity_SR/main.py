from condPhIREGANs import *

data_type = 'wind'
data_path = 'example_data/example_data.tfrecord'
model_path = 'models/trained_gan/gan'
r = [2, 5]
mu_sig=[[0.7684, -0.4575], [4.9491, 5.8441]]


if __name__ == '__main__':

    phiregans = condPhIREGANs(data_type=data_type, mu_sig=mu_sig, save_every=1, print_every=1)

    model_dir = phiregans.train(r=r,
                                data_path=data_path,
                                model_path=None,
                                batch_size=3,
                                rep_size=5)
    
    phiregans.test(r=r,
                   data_path=data_path,
                   model_path=model_path,
                   batch_size=1,
                   rep_size=10,
                   plot_data=True)



