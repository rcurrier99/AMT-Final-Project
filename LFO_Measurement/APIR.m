%--------------------------------------------------------------------------
% Ry Currier
% 2022-07-04
% 
% Cascaded 1st-Order Allpass Filter Impulse Response Generator
%--------------------------------------------------------------------------

clc; clear all; close all;

%% Params

a = -0.9;                                       % Filter Coefficient
N = 64;                                         % Number of Cascaded Filters

%% Derived Params

tend = (a^2 - 1) / (a^2 - 2*a + 1);             % End Frequency Group Delay
tstart = (a^2 - 1) / (a^2 + 2*a + 1);           % Start Frequency Group Delay
T = floor((tend - tstart) * N);                 % IR Duration (samps)

%% Impulse Response

u = [1, zeros(1, T-1)];                         % Unit Impulse Vector

for j=1:N
    u = filter([a, 1], [1, a], u);
end

%% Measurement Signal

Fs = 44.1e3;                                    % Sample Rate
Nf = floor(20*Fs);                              % Measurement Signal Length (samps)
x = zeros(1, Nf);                               % Initial Measurement Signal
s = 0.001;                                       % Chirp Spacing (s)
S = floor(s*Fs);                                % Chirp Spacing (samps)
d = floor(Nf / (T+S));                          % Number of Chirps

% Loop over Number of Chirps
for i=1:d
    
    x(1+(i-1)*(T+S):i*(T+S)) = [u,zeros(1,S)];  % Put Chirps in Measurement Signal
    
end

%% Write to Files

audiowrite("Chirp.wav", u, Fs)
audiowrite("VAML_Phasor_Measurement_Signal.wav", x, Fs)
