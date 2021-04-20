import os
import json
import argparse
import numpy as np
from opendr.topology import loop_subdivider
from psbody.mesh import Mesh

def custom_obj_save_with_mtl(file_dir, base_name, file_name, vertices, faces, vts):
    with open('./obj_info.txt', 'r') as f_info:
        info = f_info.readlines()
    with open('./mtl_info.txt', 'r') as m_info:
        mtl_info = m_info.readlines()
    vt_info = [i for i in info if i.startswith('vt')]
    file_dir = os.path.join(file_dir, file_name)
    with open(file_dir, 'w') as fp:
        fp.write('mtlib %s\n'%file_name.replace('.obj','.mtl'))
        for v in vertices:
            fp.write('v {:f} {:f} {:f}\n'.format(v[0], v[1], v[2]))
        for f, vt, vn in faces:
            fp.write('f {:d}/{:d}/{:d} {:d}/{:d}/{:d} {:d}/{:d}/{:d}\n'.format(f[0], vt[0], vn[0], f[1], vt[1], vn[1], f[2], vt[2], vn[2],))
        fp.write(''.join(vt_info))
        fp.write('\nusemtl Mymtl')
    with open(file_dir.replace('.obj', '.mtl'), 'w') as fm:
        fm.write(''.join(mtl_info))
        fm.write('map_Kd %s_octopus_hres.jpg' % (base_name))

def upsampling(dataroot, base_name, mesh_list):
    with open('./mtl_info.txt', 'r') as m_info:
        mtl_info = m_info.readlines()
    for mesh in mesh_list:
        body = Mesh(filename=os.path.join(dataroot, mesh))
        v, f = body.v, body.f
        (mapping, hf) = loop_subdivider(v, f)
        hv = mapping.dot(v.ravel()).reshape(-1, 3)
        body_hres = Mesh(hv, hf)

        vt, ft = np.hstack((body.vt, np.ones((body.vt.shape[0], 1)))), body.ft
        (mappingt, hft) = loop_subdivider(vt, ft)
        hvt = mappingt.dot(vt.ravel()).reshape(-1, 3)[:, :2]
        body_hres.vt, body_hres.ft = hvt, hft

        #body_hres.set_texture_image(body_tex)
        body_hres.write_obj(os.path.join(dataroot, mesh.replace('.obj', '_hres.obj')))

        with open(os.path.join(dataroot, mesh.replace('.obj', '_hres.obj')), 'r') as original: data = original.read()
        with open(os.path.join(dataroot, mesh.replace('.obj', '_hres.obj')), 'w') as modified: modified.write(f"mtllib {mesh.replace('.obj', '_hres.mtl')}\n" + data)

        with open(os.path.join(dataroot, mesh.replace('.obj', '_hres.mtl')), 'w') as fm:
            fm.write(''.join(mtl_info))
            fm.write('map_Kd %s_octopus_hres.jpg' % (base_name))


        #custom_obj_save_with_mtl(dataroot, base_name, mesh, hv, hf, hvt, hvt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'dataroot',
        type=str,
        default='../data/Merging_Splitting/1_0',
        help="Dataset dir")

    args = parser.parse_args()
    dataroot = args.dataroot
    base_name = os.path.basename(dataroot)
    mesh_list = [f'{base_name}_with_garment.obj', f'{base_name}_splited_up.obj', f'{base_name}_splited_down.obj']
    upsampling(dataroot, base_name, mesh_list)