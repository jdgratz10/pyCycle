import sys

import numpy as np

# protection incase env doesn't have matplotlib installed, since its not strictly required
try: 
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError: 
  plt = None


def print_e_flow_station(prob, fs_names, file=sys.stdout):
    names = ['tot:P', 'tot:T', 'tot:h', 'tot:S', 'stat:P', 'stat:W', 'stat:MN', 'stat:V', 'stat:area']

    n_names = len(names)
    line_tmpl = '{:<23}|  '+'{:>13}'*n_names
    len_header = 27+13*n_names

    print("-"*len_header, file=file, flush=True)
    print("                            FLOW STATIONS", file=file, flush=True)
    print("-"*len_header, file=file, flush=True)

    # header_line
    vals = ['Flow Station'] + names
    print('-'*len_header, file=file, flush=True)
    print(line_tmpl.format(*vals), file=file, flush=True)
    print('-'*len_header, file=file, flush=True)


    line_tmpl = '{:<23.23}|  ' + '{:13.3f}'*n_names
    for fs_name in fs_names:
        data = []
        for name in names:
            full_name = '{}:{}'.format(fs_name, name)
            data.append(prob[full_name][0])

        vals = [fs_name] + data
        print(line_tmpl.format(*vals), file=file, flush=True)
    print('-'*len_header, file=file, flush=True)


def print_e_compressor(prob, element_names, file=sys.stdout):

    len_header = 17+14*13
    # print("-"*len_header)
    print("-"*len_header, file=file, flush=True)
    print("                          COMPRESSOR PROPERTIES", file=file, flush=True)
    print("-"*len_header, file=file, flush=True)

    line_tmpl = '{:<14}|  '+'{:>11}'*14
    print(line_tmpl.format('Compressor', 'Wc', 'Pr', 'eta_a', 'eta_p', 'Nc', 'pwr', 'RlineMap', 'NcMap', 'WcMap', 'PRmap', 'alphaMap', 'SMN', 'SMW', 'effMap'),
          file=file, flush=True)
    print("-"*len_header, file=file, flush=True)


    line_tmpl = '{:<14}|  '+'{:11.3f}'*14
    for e_name in element_names:
        sys = prob.model._get_subsystem(e_name)
        if sys.options['design']:
          PR_temp = prob[e_name+'.map.scalars.PR'][0]
          eff_temp = prob[e_name+'.map.scalars.eff'][0]
        else:
          PR_temp = prob[e_name+'.PR'][0]
          eff_temp = prob[e_name+'.eff'][0]

        print(line_tmpl.format(e_name, prob[e_name+'.Wc'][0], PR_temp,
                               eff_temp, prob[e_name+'.eff_poly'][0], prob[e_name+'.Nc'][0], prob[e_name+'.power'][0],
                               prob[e_name+'.map.RlineMap'][0], prob[e_name+'.map.NcMap'][0],prob[e_name+'.map.PRmap'][0], prob[e_name+'.map.WcMap'][0],
                               prob[e_name+'.map.map.alphaMap'][0], prob[e_name+'.SMN'][0], prob[e_name+'.SMW'][0], prob[e_name+'.map.effMap'][0]),
              file=file, flush=True)
    print("-"*len_header, file=file, flush=True)




def print_e_nozzle(prob, element_names, file=sys.stdout):

    len_header = 17+8*13
    print("-"*len_header, file=file, flush=True)
    print("                            NOZZLE PROPERTIES", file=file, flush=True)
    print("-"*len_header, file=file, flush=True)

    line_tmpl = '{:<14}|  '+'{:>13}'*8
    print(line_tmpl.format('Nozzle', 'PR', 'Cv', 'Cfg', 'Ath', 'MNth', 'MNout', 'V', 'Fg'), file=file, flush=True)


    for e_name in element_names:
        sys = prob.model._get_subsystem(e_name)
        if sys.options['lossCoef'] == 'Cv':
            Cv_val = prob[e_name+'.Cv'][0]
            Cfg_val = '        N/A  '
            line_tmpl = '{:<14}|  ' + '{:13.3f}'*2 + '{}' + '{:13.3f}'*5

        else:
            Cv_val = '        N/A  '
            Cfg_val = prob[e_name+'.Cfg'][0]
            line_tmpl = '{:<14}|  ' + '{:13.3f}'*1 + '{}' + '{:13.3f}'*6

        print(line_tmpl.format(e_name, prob[e_name+'.PR'][0], Cv_val, Cfg_val,
                               prob[e_name+'.Throat:stat:area'][0], prob[e_name+'.Throat:stat:MN'][0],
                               prob[e_name+'.Fl_O:stat:MN'][0],
                               prob[e_name+'.Fl_O:stat:V'][0], prob[e_name+'.Fg'][0]),
             file=file, flush=True)



def print_e_shaft(prob, element_names, file=sys.stdout):

    len_header = len_header = 23+20*5

    print("-"*len_header, file=file, flush=True)
    print("                            SHAFT PROPERTIES", file=file, flush=True)
    print("-"*len_header, file=file, flush=True)

    line_tmpl = '{:<20}|  '+'{:>20}'*5
    print(line_tmpl.format('Shaft', 'Nmech', 'trqin', 'trqout', 'pwrin', 'pwrout'), file=file)

    line_tmpl = '{:<20}|  '+'{:20.3f}'*5
    for e_name in element_names:
        print(line_tmpl.format(e_name, prob[e_name+'.Nmech'][0],
                               prob[e_name+'.trq_in'][0],
                               prob[e_name+'.trq_out'][0],
                               prob[e_name+'.pwr_in'][0],
                               prob[e_name+'.pwr_out'][0]),
              file=file, flush=True)



def plot_e_compressor_maps(prob, element_names, eff_vals=np.array([0,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]),alphas=[0]):

    for e_name in element_names:
        comp = prob.model._get_subsystem(e_name)
        map_data = comp.options['map_data']

        s_Wc = prob[e_name+'.s_Wc']
        s_PR = prob[e_name+'.s_PR']
        s_eff = prob[e_name+'.s_eff']
        s_Nc = prob[e_name+'.s_Nc']

        RlineMap, NcMap = np.meshgrid(map_data.RlineMap, map_data.NcMap, sparse=False)

        for alpha in alphas:
          scaled_PR = (map_data.PRmap[alpha,:,:] - 1.) * s_PR + 1.

          plt.figure(figsize=(11,8))
          Nc = plt.contour(map_data.WcMap[alpha,:,:]*s_Wc,scaled_PR,NcMap*s_Nc,colors='k',levels=map_data.NcMap*s_Nc)
          R = plt.contour(map_data.WcMap[alpha,:,:]*s_Wc,scaled_PR,RlineMap,colors='k',levels=map_data.RlineMap)
          eff = plt.contourf(map_data.WcMap[alpha,:,:]*s_Wc,scaled_PR,map_data.effMap[alpha,:,:]*s_eff,levels=eff_vals)

          plt.colorbar(eff)

          plt.plot(prob[e_name+'.Wc'], prob[e_name+'.map.scalars.PR'][0], 'ko')

          plt.clabel(Nc, fontsize=9, inline=False)
          plt.clabel(R, fontsize=9, inline=False)
          # plt.clabel(eff, fontsize=9, inline=True)

          plt.xlabel('Wc, lbm/s')
          plt.ylabel('PR')
          plt.title(e_name)
          # plt.show()
          plt.savefig(e_name+'.pdf') 