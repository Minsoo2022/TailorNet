import os
import json
import argparse
from psbody.mesh import Mesh
from utils.interpenetration import remove_interpenetration_fast, remove_interpenetration_fast_custom

# Set output path where inference results will be stored

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--dataroot", default="../data/BCNet_merging_mesh/6_0/")
    opt = parser.parse_args()
    return opt

def custom_obj_save(file_dir, file_name, base_name, object):
    file_dir = os.path.join(file_dir, file_name)
    with open('./obj_info.txt', 'r') as f_info:
        info = f_info.readlines()
    with open('./mtl_info.txt', 'r') as m_info:
        mtl_info = m_info.readlines()
    with open(file_dir.replace('.obj', '.mtl'), 'w') as fm:
        fm.write(''.join(mtl_info))
        fm.write('map_Kd %s.jpg' % (base_name))
    with open(file_dir, 'w') as fp:
        for v in object.v:
            fp.write('v {:f} {:f} {:f}\n'.format(v[0], v[1], v[2]))
        fp.write(''.join(info))
        fp.write('\nusemtl Mymtl')

def convert(o):
    return int(o)


def make_body_mesh_with_clothes(opt):
    #file_dir = '../data/BCNet_merging_mesh/4_0/'
    file_dir = opt.dataroot
    file_name = [i for i in file_dir.split('/') if len(i)>0][-1]
    body = Mesh(filename=os.path.join(file_dir,f'{file_name}_ori.obj'))
    up = Mesh(filename=os.path.join(file_dir,f'{file_name}_up.obj'))
    bottom = Mesh(filename=os.path.join(file_dir,f'{file_name}_bottom.obj'))

    up = remove_interpenetration_fast(up, body)
    pred_body, indices_up = remove_interpenetration_fast_custom(body, up, threshold=0.0010)
    pred_body.write_obj(os.path.join(file_dir, 'deformed_body_with_shirt.obj'))

    bottom = remove_interpenetration_fast(bottom, pred_body)
    pred_body, indices_down = remove_interpenetration_fast_custom(pred_body, bottom, threshold=0.0005)
    pred_body.write_obj(os.path.join(file_dir, 'deformed_body_with_pant_shirt.obj'))
    pred_body.write_obj(os.path.join(file_dir, f'{file_name}_with_garment.obj'))
    custom_obj_save(file_dir, f'{file_name}_with_garment.obj', file_name, pred_body)
    indices_info = {'indices_down' : [int(i) for i in list(indices_down)],
                    'indices_up' : [int(i) for i in list(indices_up)]}
    with open(os.path.join(file_dir, 'indices_info.json'), 'w') as json_file:
        json.dump(indices_info, json_file)




if __name__ == '__main__':
    opt= get_opt()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    make_body_mesh_with_clothes(opt)
