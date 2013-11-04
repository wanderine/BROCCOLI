function filelist = editfilenames(filelist,varargin)
% Prepend, append or change extension of a list of filenames
% FUNCTION filelist = editfilenames(filelist,action,fix, action,fix,...)
%   filelist - char array or a cell array of strings.
%   action   - either 'prefix', 'suffix' or 'ext'
%   fix      - corresponding string

if rem(length(varargin),2)
	error('Not enough input arguments.');
end

if ischar(filelist), filelist = cellstr(filelist); end

for i=1:2:length(varargin)
	chext = 0;
	switch varargin{i}
		case 'prefix'
			prefix = varargin{i+1};
			suffix = '';
		case 'suffix'
			prefix = '';
			suffix = varargin{i+1};
		case 'ext'
			prefix = '';
			suffix = '';
			chext = varargin{i+1};
		otherwise
			error('Unknown action.');
	end

	for j=1:numel(filelist)
		[pth,nam,ext,num] = spm_fileparts(deblank(filelist{j}));
		if ischar(chext), ext = chext; end
		if iscell(prefix), prfx = prefix{j}; else prfx = prefix; end
		if iscell(suffix), sfix = suffix{j}; else sfix = suffix; end
		filelist{j}  = fullfile(pth,[prfx nam sfix ext num]);
	end
end