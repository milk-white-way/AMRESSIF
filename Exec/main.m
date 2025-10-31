close all;
clearvars;

[P.t, P.x, P.y, P.z, P.u] = importfile('ren1000_vertical_numel_positive100.txt');
[N.t, N.x, N.y, N.z, N.u] = importfile('ren1000_vertical_numel_negative100.txt');

z = (N.z + P.z)/2;

if z(1) == 0.5
    y = N.y;
    u = (N.u + P.u)/2;
    plot(y, u, 'o');
end
