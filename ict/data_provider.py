from collagen.data import ItemLoader, MixUpSampler, DataProvider


def ict_data_provider(model, alpha, n_classes, train_labeled_data, train_unlabeled_data, val_labeled_data, val_unlabeled_data,
                      transforms, parse_item, bs, num_threads, item_loaders=dict(), root=""):
    """
    Default setting of data provider for ICT
    """
    item_loaders["labeled_train"] = ItemLoader(meta_data=train_labeled_data, name='l_norm',
                                                 transform=transforms[1], parse_item_cb=parse_item, batch_size=bs,
                                                 num_workers=num_threads, root=root, shuffle=True)

    item_loaders["unlabeled_train"] = MixUpSampler(meta_data=train_unlabeled_data, name='u_mixup', alpha=alpha, model=model,
                                                 transform=transforms[0], parse_item_cb=parse_item, batch_size=bs,
                                                 num_workers=num_threads, root=root, shuffle=True)

    item_loaders["labeled_eval"] = ItemLoader(meta_data=val_labeled_data, name='l_norm',
                                                 transform=transforms[1], parse_item_cb=parse_item, batch_size=bs,
                                                 num_workers=num_threads, root=root, shuffle=False)

    item_loaders["unlabeled_eval"] = MixUpSampler(meta_data=val_unlabeled_data, name='u_mixup', alpha=alpha, model=model,
                                                 transform=transforms[1], parse_item_cb=parse_item, batch_size=bs,
                                                 num_workers=num_threads, root=root, shuffle=False)

    return DataProvider(item_loaders)
