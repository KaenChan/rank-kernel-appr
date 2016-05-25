function [dataset_txt, dataset_mat] = get_dataset_name(s)
    switch lower(s)
        case 'td2004'
            dataset_pc   = 'data/TD2004';
            dataset_mat  = 'letor2_td2004_fold';
        case 'ohsumed'
            dataset_pc   = 'data/OHSUMED';
            dataset_mat  = 'letor3_ohsumed_fold';
        case 'mq2007'
            dataset_pc   = 'data/MQ2007';
            dataset_mat  = 'letor4_mq2007_fold';
        case 'mq2008'
            dataset_pc   = 'data/MQ2008';
            dataset_mat  = 'letor4_mq2008_fold';
    end

    dataset_txt = dataset_pc;
    end
