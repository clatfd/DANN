import numpy as np
import os
import cv2
import pickle

def collectsegpatch(seqbbsim, seqid, seqbbid, neigh, pi, segpatchfolder, polaravail):
    slicei = seqbbsim[seqid][seqbbid][0]
    bbid = seqbbsim[seqid][seqbbid][1]
    if polaravail:
        patchfilename = segpatchfolder + '/' + pi + '_%03d_%d_%d.png' % (slicei, seqid, bbid)
    else:
        patchfilename = segpatchfolder + '/cart_' + pi + '_%03d_%d_%d.png' % (slicei, seqid, bbid)
    if not os.path.exists(patchfilename):
        print('no patch', patchfilename)

    segpatchfilenamelist = []
    segpatchfilenamelist.append(patchfilename)
    cslicei = seqbbsim[seqid][seqbbid][0]
    for neighi in range(1, neigh + 1):
        # prepend
        pseqbbid = seqbbid - neighi
        if pseqbbid >= 0:
            pslicei, pbbid = seqbbsim[seqid][pseqbbid]
            if cslicei - pslicei == neighi:
                if polaravail:
                    ppatchfilename = segpatchfolder + '/' + pi + '_%03d_%d_%d.png' % (pslicei, seqid, pbbid)
                else:
                    ppatchfilename = segpatchfolder + '/cart_' + pi + '_%03d_%d_%d.png' % (pslicei, seqid, pbbid)
            else:
                ppatchfilename = segpatchfilenamelist[0]
            assert os.path.exists(ppatchfilename)
            segpatchfilenamelist.insert(0, ppatchfilename)
        else:
            segpatchfilenamelist.insert(0, segpatchfilenamelist[0])
        # append
        nseqbbid = seqbbid + neighi
        if nseqbbid < len(seqbbsim[seqid]):
            nslicei, nbbid = seqbbsim[seqid][nseqbbid]
            if nslicei - cslicei == neighi:
                if polaravail:
                    npatchfilename = segpatchfolder + '/' + pi + '_%03d_%d_%d.png' % (nslicei, seqid, nbbid)
                else:
                    npatchfilename = segpatchfolder + '/cart_' + pi + '_%03d_%d_%d.png' % (nslicei, seqid, nbbid)
            else:
                npatchfilename = segpatchfilenamelist[-1]
            assert os.path.exists(npatchfilename)
            segpatchfilenamelist.append(npatchfilename)
        else:
            segpatchfilenamelist.append(segpatchfilenamelist[-1])
    # classfication CNN input size
    segstack = np.zeros((64, 64, 5, 2))
    for segnameid in range(len(segpatchfilenamelist)):
        segimg = cv2.imread(segpatchfilenamelist[segnameid])
        segstack[:, :, segnameid, :] = segimg[segimg.shape[0] // 2 - 32:segimg.shape[0] // 2 + 32,
                                       segimg.shape[1] // 2 - 32:segimg.shape[1] // 2 + 32, :2] / 255
    return segstack


def data_generator(pilist, config, batch_size, load_gt=True):
    segpatch_batch = np.zeros((batch_size, 64, 64, 5, 2))
    seglabel_batch = np.zeros((batch_size, 1))
    neigh = 2
    polaravail = True
    mi = 0
    while 1:
        for pi in pilist[:]:
            listfilename = config['cdir'] +'/3Dmergelist/'+pi+'/'+pi+'.list'
            if not os.path.exists(listfilename):
                print('No list file created', listfilename)
                return
            with open(listfilename, 'rb') as fp:
                all_artery_objs = pickle.load(fp)

            seqbbfilename = config['cdir'] + '/3Dmergeseqbb/' + pi + '/' + pi + '.seqbb.npy'
            if os.path.exists(seqbbfilename):
                seqbbsim = np.load(seqbbfilename, allow_pickle=True)
            else:
                print('no seqbb', seqbbfilename)

            segpatchfolder = config['cdir'] +'/3Dmergesegpatch/' +pi+'/'

            seqlabelfilename = config['cdir'] + '/3Dmergeseqbb/' + pi + '/' + pi + '.seqlabel.npy'
            if os.path.exists(seqlabelfilename):
                seqlabel = np.load(seqlabelfilename, allow_pickle=True)
            else:
                print('no seqlabel')
            print(pi)
            for seqid in range(len(seqbbsim)):
                if seqlabel[seqid] == -1:
                    continue
                for seqbbid in range(len(seqbbsim[seqid])):
                    if load_gt:
                        slicei, bbid = seqbbsim[seqid][seqbbid]
                        artlabel = all_artery_objs[slicei]['object'][bbid]['artlabel']
                        if not artlabel in all_artery_objs[slicei]['gtlabel']:
                            continue
                        gtlabel_type = all_artery_objs[slicei]['gtlabel'][artlabel]
                        # gtlabel 0(unqualify)
                        if gtlabel_type == 0:
                            continue

                    segpatch = collectsegpatch(seqbbsim, seqid, seqbbid, neigh, pi, segpatchfolder, polaravail)

                    segpatch_batch[mi] = segpatch

                    if load_gt:
                        # cnn target -1, 0, 1, gtlabel 0(unqualify) 1,2,3
                        seglabel_batch[mi] = gtlabel_type - 2

                    if mi == batch_size - 1:
                        mi = 0
                        yield segpatch_batch, seglabel_batch
                    else:
                        mi += 1
                    # print(slicei, bbid,gtlabel_type)


def load_patch(pilist, config, skip=1, load_gt=True, load_src=False):
    neigh = 2
    polaravail = True
    segpatch_batch = []
    seglabel_batch = []
    segsrc_batch = []
    ct = 0
    for pi in pilist[:]:

        listfilename = config['cdir'] + '/3Dmergelist/' + pi + '/' + pi + '.list'
        if not os.path.exists(listfilename):
            print('No list file created', listfilename)
            return
        with open(listfilename, 'rb') as fp:
            all_artery_objs = pickle.load(fp)

        seqbbfilename = config['cdir'] + '/3Dmergeseqbb/' + pi + '/'+ pi + '.seqbb.npy'
        if os.path.exists(seqbbfilename):
            seqbbsim = np.load(seqbbfilename, allow_pickle=True)
        else:
            print('no seqbb', seqbbfilename)

        segpatchfolder = config['cdir'] +'/3Dmergesegpatch/'+pi+'/'

        seqlabelfilename = config['cdir'] + '/3Dmergeseqbb/' + pi + '/' + pi + '.seqlabel.npy'
        if os.path.exists(seqlabelfilename):
            seqlabel = np.load(seqlabelfilename, allow_pickle=True)
        else:
            print('no seqlabel')

        #print(pi)
        for seqid in range(len(seqbbsim)):
            if seqlabel[seqid] == -1:
                continue
            for seqbbid in range(len(seqbbsim[seqid])):
                if load_gt:
                    slicei, bbid = seqbbsim[seqid][seqbbid]
                    #print(latte.all_artery_objs[slicei]['object'],bbid)
                    artlabel = all_artery_objs[slicei]['object'][bbid]['artlabel']
                    if not artlabel in all_artery_objs[slicei]['gtlabel']:
                        continue
                    gtlabel_type = all_artery_objs[slicei]['gtlabel'][artlabel]
                    if gtlabel_type == 0:
                        continue
                ct += 1

                if ct%skip != 0:
                    continue

                segpatch = collectsegpatch(seqbbsim, seqid, seqbbid, neigh, pi, segpatchfolder, polaravail)

                segpatch_batch.append(segpatch)

                if load_gt:
                    # cnn target -1, 0, 1
                    seglabel_batch.append(gtlabel_type - 2)

                if load_src:
                    segsrc_batch.append({'pi':pi,'seqid':seqid,'seqbbid':seqbbid})

    if load_src:
        return np.array(segpatch_batch), np.array(seglabel_batch), segsrc_batch
    else:
        return np.array(segpatch_batch), np.array(seglabel_batch)
