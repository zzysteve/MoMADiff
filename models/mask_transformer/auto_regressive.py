import torch

class MaskGeneration(object):
    def __init__(self, lengths, max_len, num_steps=16, noise_schedule=None, device='cuda', drop_last=True, ignored_position=None, permutation_mode='random'):
        assert len(lengths.shape) == 1
        self.lengths = lengths
        self.max_len = max_len
        self.noise_schedule = noise_schedule
        self.num_steps = num_steps
        self.n_batch = len(lengths)
        self.current_step = 0
        
        # Note that this only affect __init__ method. By creating an extra step if drop_last is True, the last step is ignored in the __next__ method.
        if drop_last:
            num_steps += 1
        
        self.perm = torch.zeros((self.n_batch, max_len), device=device, dtype=torch.long)
        for i, length in enumerate(lengths):
            # Larger values will be generate at the first iteration
            if permutation_mode == 'random':
                self.perm[i, :length] = torch.randperm(length, device=device)
            elif permutation_mode == 'linear':
                self.perm[i, :length] = torch.arange(length-1, -1, -1, device=device)
            elif permutation_mode == 'coarse2fine':
                # # generate an cosine function with the interval of 20, aborted due some action may less than 20
                # interval = 20
                # self.perm[i, :length] = torch.ceil((torch.cos(torch.linspace(0, length, length, device=device) * 2 * torch.pi / interval) + 1) * length) - 1
                
                # generate a cosine function with 10 key points
                key_points = 5
                self.perm[i, :length] = torch.floor((torch.cos((2 * torch.pi * key_points / length) * torch.linspace(0, length, length, device=device)) + 1) * (length - 1) / 2.0)
                # print(self.perm[i, :length])
            elif permutation_mode == 'two_way_linear':
                # generate a V-shape permutation, for example, 3, 2, 1, 0, 1, 2, 3
                self.perm[i, :length] = torch.cat((torch.arange(length-1, -1, -2, device=device), torch.arange(0 , length-1, 2, device=device)))
            else:
                raise ValueError(f'In MaskGeneration: invalid permutation mode {permutation_mode}')
            self.perm[i, length:] = max_len + 1
            # mark the ignored position
            if ignored_position is not None:
                self.perm[i, ignored_position[i]] = max_len + 1
        # print(self.perm)
        # another step method, not using for now
        if noise_schedule == None:
            schedule = torch.linspace(0, 1, num_steps, device=device).unsqueeze(-1)
        else:
            schedule = self.noise_schedule(torch.linspace(0, 1, num_steps, device=device)).unsqueeze(-1)
        self.mask_table = torch.floor(schedule * lengths)
        # print("[Debug] self.perm: ", self.perm[0])
        # print("[Debug] schedule: ", schedule[0])
        # print("[Debug] mask_table: ", self.mask_table[0])

    def __iter__(self):
        return self

    def __next__(self):
        '''
        Generate mask for each token
        :param t: int, current time step
        '''
        if self.current_step >= self.num_steps:
            raise StopIteration
        mask = (self.perm < self.mask_table[self.current_step].unsqueeze(-1))
        self.current_step += 1
        return mask

if __name__ == '__main__':
    mask_gen = MaskGeneration(torch.Tensor((4,6,5,8)), 8, 5, device='cpu')
    for mask in mask_gen:
        print(mask)