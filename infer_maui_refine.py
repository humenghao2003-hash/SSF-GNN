"""Refine existing Wildfire tiles with overlapping 128x128 windows.
Each 128x128 crop is resized to 256x256 because the released model is fixed
at its training input size; the output is a fine-scale probability map.
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from osgeo import gdal
from Compute_indices import compute_indices
from dataset import class_dict
from model import SSFGNN

SOURCE = Path("data/exp/maui_wildfire_lahaina_2023-08-18_S2L2A_12band_15km.tif")
BASE = Path("data/exp/maui_wildfire_inference")
OUT = BASE / "refined_128"
TILE, SMALL, STEP = 256, 128, 64
MEAN = np.array([0.0736,0.0802,0.0935,0.0933,0.1120,0.1290,0.1386,0.1369,0.1398,0.1532,0.1195,0.0972], np.float32)
STD = np.array([0.1041,0.1055,0.1128,0.1293,0.1377,0.1508,0.1594,0.1627,0.1651,0.1768,0.1612,0.1433], np.float32)


def make_input(raw):
    refl = np.clip((raw.astype(np.float32)-1000)/10000, -0.1, 1.0)
    idx = compute_indices(refl)
    norm = (refl - MEAN[:,None,None]) / STD[:,None,None]
    return np.concatenate((norm, idx), axis=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ds = gdal.Open(str(SOURCE)); raw = ds.ReadAsArray().astype(np.float32)
    coarse = list(csv.DictReader((BASE/'tile_predictions.csv').open()))
    selected = [(int(r['row']), int(r['column'])) for r in coarse if r['prediction']=='Wildfire']
    crops, meta = [], []
    for row,col in selected:
        y0,x0=row*TILE,col*TILE
        for dy in (0,STEP,2*STEP):
            for dx in (0,STEP,2*STEP):
                crop=raw[:, y0+dy:y0+dy+SMALL, x0+dx:x0+dx+SMALL]
                x=torch.from_numpy(make_input(crop)).unsqueeze(0)
                x=F.interpolate(x, size=(256,256), mode='bilinear', align_corners=False).squeeze(0).numpy()
                crops.append(x); meta.append((row,col,dy,dx))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SSFGNN(7); model.load_state_dict(torch.load('checkpoints_new/best.pt',map_location='cpu',weights_only=True)); model.to(device).eval()
    probs=[]
    with torch.inference_mode():
        for batch in torch.from_numpy(np.stack(crops)).split(6):
            probs.append(torch.softmax(model(batch.to(device)),1).cpu().numpy())
    probs=np.concatenate(probs); pred=probs.argmax(1); conf=probs.max(1)
    with (OUT/'refined_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['parent_row','parent_col','offset_y','offset_x','prediction','confidence','wildfire_probability','all_probabilities'])
        for m,p,c in zip(meta,pred,conf): w.writerow([*m,class_dict[p],f'{c:.6f}',f'{probs[len(w._rows) if False else 0,1]:.6f}' if False else f'{probs[len(w._rows)-1,1] if False else 0:.6f}'])
    # Rewrite CSV cleanly (avoid relying on writer internals above).
    with (OUT/'refined_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['parent_row','parent_col','offset_y','offset_x','prediction','confidence','wildfire_probability','all_probabilities'])
        for m,p,c,pr in zip(meta,pred,conf,probs): w.writerow([*m,class_dict[p],f'{c:.6f}',f'{pr[1]:.6f}',';'.join(f'{v:.5f}' for v in pr)])
    # Global heatmap: max Wildfire probability over the selected windows.
    h,w=raw.shape[1],raw.shape[2]; heat=np.full((h,w),np.nan,np.float32)
    for (row,col,dy,dx),pr in zip(meta,probs):
        y,x=row*TILE+dy,col*TILE+dx; heat[y:y+SMALL,x:x+SMALL]=np.nanmax(np.dstack((np.nan_to_num(heat[y:y+SMALL,x:x+SMALL],nan=0),np.full((SMALL,SMALL),pr[1]))),axis=2)
    rgb=np.moveaxis(raw[[3,2,1]],0,-1); lo,hi=np.percentile(rgb,(1,99)); rgb=np.clip((rgb-lo)/max(hi-lo,1),0,1)
    fig,ax=plt.subplots(figsize=(12,12)); ax.imshow(rgb)
    im=ax.imshow(heat,cmap='hot',vmin=0,vmax=1,alpha=np.where(np.isnan(heat),0,0.62)); fig.colorbar(im,ax=ax,fraction=.03,label='Wildfire probability')
    for row,col in selected: ax.add_patch(plt.Rectangle((col*TILE,row*TILE),TILE,TILE,fill=False,edgecolor='cyan',linewidth=2))
    ax.set_title('128x128 refinement within coarse Wildfire tiles (resized to 256 for model)'); ax.axis('off'); fig.savefig(OUT/'refined_wildfire_heatmap.png',dpi=220,bbox_inches='tight'); plt.close(fig)
    print('Selected coarse tiles:',selected); print('Refined windows:',len(meta)); print('Refined Wildfire predictions:',int((pred==1).sum())); print('Max wildfire probability:',float(probs[:,1].max()))

if __name__=='__main__': main()
