%--------------------------------------------------------------------------
% Ry Currier
% 2022-07-04
% 
% MXR Phase 90 LFO Fitting 
%--------------------------------------------------------------------------

clc; clear all; close all;

%% Load Files

n = 3;                                          % File Number
file1 = "PHASE_Measurement_" + num2str(n) + ".wav";
file2 = "Chirp.wav";

[x, Fs] = audioread(file1);
x = x(1:20*Fs);

[chirp, ~] = audioread(file2);

%% Important Constants

T = length(chirp);                              % Length of Chirp (samps)
L = length(x);                                  % Length of Measured Signal (samps)
S = floor(0.001*Fs);                            % Chirp Spacing (samps)
d = floor(L / (T+S));                           % Number of Chirps
F = floor(Fs / 6);                              % Notch Frequency Range

%% FFTs

spec = zeros(F,d);                              % Matrix for Storing FFTs of each Chirp

for i=1:d
    chirp = x((T+S)*(i-1)+1:(T+S)*(i-1)+T);     % Isolate Chirp
    ffti = fft(chirp,Fs);                       % Take FFT
    spec(:,i) = ffti(1:F,:);                    % Add to Matrix
end

spec = abs(spec);                               % Magnitude of Spectrum
spec = spec/max(max(spec));                     % Normalize

%% Plot

t = [1:d]*(T+S)/Fs;                             % Time Vector
f = [1:F];                                      % Frequency Vector
imagesc(t,f,1-spec)                             % Attend to Notch Path
set(gca,'YDir','normal')
colorbar
xlabel("Time(s)")
ylabel("Frequency (Hz)")

%% Fit LFO

% Initial Parameter Guess
LFOom = [0.15; 0.3; 1.2; 2; 5] * 2*pi;
LFOphi = [1; 1; -0.5; 0.5; 0.5] * pi;
LFOA = -6500;
LFOB = 7000;

% LFO Function
LFOfn = @(b,x) b(3)*abs(sin(b(1)/2*x + b(2)/2)) + b(4);     

rng('default')

[M,I] = max(1-spec);
beta0 = [LFOom(n); LFOphi(n); LFOA; LFOB];            % Initial Guess Vector
beta = nlinfit(t,I,LFOfn,beta0);                 % Nonlinear Fit

LFO1 = LFOfn(beta,t);                            % LFO

% Cook the fuck outta them books
for i=1:d
    j=1;
    while j<LFO1(i)
        spec(j,i) = 1;
        j=j+1;
    end
end

[M,I] = max(1-spec);
beta1 = beta;                                   % Second Guess Vector
beta = nlinfit(t,I,LFOfn,beta1);                 % Nonlinear Fit

LFO2 = LFOfn(beta,t);                            % LFO

hold on
plot3(t,LFO2,ones(length(t)),'LineWidth',2)        % Plot LFO on Spectrogram
hold off
