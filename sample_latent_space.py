import os

import click
import torch
from psbody.mesh import Mesh, MeshViewers
from torch_geometric.data import DataLoader
from tqdm import trange

import mesh_operations
from data import ComaDataset
from main import read_config, scipy_to_torch_sparse
from model import Coma
from transform import Normalize


@click.command()
@click.option('--checkpoint', default='checkpoint/checkpoint_293.pt')
@click.option('--config-path', default='default.cfg')
@click.option('--output-dir', default='movie')
def main(checkpoint, config_path, output_dir):
    config = read_config(config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Initializing parameters')
    template_file_path = config['template_fname']
    template_mesh = Mesh(filename=template_file_path)

    print('Generating transforms')
    M, A, D, U = mesh_operations.generate_transform_matrices(
        template_mesh, config['downsampling_factors'])

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]

    print('Preparing dataset')
    data_dir = config['data_dir']
    normalize_transform = Normalize()
    dataset = ComaDataset(data_dir,
                          dtype='test',
                          split='sliced',
                          split_term='sliced',
                          pre_transform=normalize_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    print('Loading model')
    model = Coma(dataset, config, D_t, U_t, A_t, num_nodes)
    checkpoint = torch.load(checkpoint)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    print('Generating latent')
    data = next(iter(loader))
    with torch.no_grad():
        data = data.to(device)
        x = data.x.reshape(data.num_graphs, -1, model.filters[0])
        z = model.encoder(x)

    print('View meshes')
    meshviewer = MeshViewers(shape=(1, 1))
    for feature_index in range(z.size(1)):
        j = torch.range(-4, 4, step=0.1, device=device)
        new_z = z.expand(j.size(0), z.size(1)).clone()
        new_z[:, feature_index] *= 1 + 0.3 * j

        with torch.no_grad():
            out = model.decoder(new_z)
            out = out.detach().cpu() * dataset.std + dataset.mean

        for i in trange(out.shape[0]):
            mesh = Mesh(v=out[i], f=template_mesh.f)
            meshviewer[0][0].set_dynamic_meshes([mesh])

            f = os.path.join(output_dir, 'z{}'.format(feature_index),
                             '{:04d}.png'.format(i))
            os.makedirs(os.path.dirname(f), exist_ok=True)
            meshviewer[0][0].save_snapshot(f, blocking=True)


if __name__ == '__main__':
    main()
