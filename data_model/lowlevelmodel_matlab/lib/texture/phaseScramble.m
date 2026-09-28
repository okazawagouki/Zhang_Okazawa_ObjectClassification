function outimg = phaseScramble(img, coh)
% outimg = phaseScramble(img, coh)
%  outimg = scrambled image
%  coh = relative balance of random phase and original phase (0.. completely random, 1 .. completely original)

    if ~exist('coh', 'var')
        coh = 0;
    end

    c = class(img);
    if isequal(c, 'uint8')
        img = double(img);
    end
    
    mask_fft = fft2(img);
    
    dims = size(mask_fft);
    
    mask_avgamp = mean(abs(mask_fft),3);

    mask_phase = angle(mask_fft);

    rand_phase = angle(fft2(rand(dims(1), dims(2))));
    new_phase = mask_phase*coh + rand_phase*(1-coh);
    new_mask_fft = mask_avgamp.*exp(1j*new_phase);
    outimg = real(ifft2(new_mask_fft));
    
    if isequal(c, 'uint8')
        outimg = uint8(outimg);
    end
end