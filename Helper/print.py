import time

def message(batch_loss , train_loss, valid_loss, iteration, iter_save, rate, epoch, start_timer, mode='print'):
    asterisk = ' '

    if mode == ('print'):
        loss = batch_loss
    if mode == ('log'):
        loss = train_loss
        if (iteration % iter_save == 0): asterisk = '*'

    text =('%0.2e   %08d%s %6.2f | ' % (rate, iteration, asterisk, epoch,)).replace('e-0', 'e-').replace('e+0',
                                                                                                       'e+') + \
        '%4.3f  %4.3f  %4.4f  | ' % (*valid_loss,) + \
        '%4.3f  %4.3f  %4.3f  | ' % (*loss,) + \
          '%s %s' % (str(time.time() - start_timer), 'min')

    return text