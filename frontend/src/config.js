const rawUrl = import.meta.env.VITE_API_URL || 'https://plant-disease-detection-r3c1.onrender.com/api';
export const API_URL = rawUrl.trim().replace(/[\r\n]/g, '').replace(/\/+$/, '');
