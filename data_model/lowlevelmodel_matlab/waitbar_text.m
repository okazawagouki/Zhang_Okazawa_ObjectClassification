function wb = waitbar_text(x, wb)
%WAITBAR_TEXT Text-based progress indicator for the command window.
%   WB = WAITBAR_TEXT(0) initializes the indicator and returns a handle.
%   WAITBAR_TEXT(FRAC, WB) updates it, FRAC in [0 1].
%   WAITBAR_TEXT('close', WB) finishes the line.
%
%   Self-contained replacement for the external waitbar_text utility used
%   by the original scripts (no GUI, safe for batch/remote runs).

    if nargin < 1 || isempty(x)
        x = 0;
    end

    if ischar(x) || isstring(x)
        if strcmpi(x, 'close')
            fprintf('\n');
        end
        wb = [];
        return
    end

    if nargin < 2 || ~isstruct(wb)
        wb = struct('len', 0);
    end

    msg = sprintf('Progress: %5.1f%%', 100 * max(0, min(1, x)));
    fprintf([repmat('\b', 1, wb.len) '%s'], msg);
    wb.len = length(msg);
end
