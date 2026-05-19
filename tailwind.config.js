/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './web_app/templates/**/*.html',
    './web_app/static/js/**/*.js',
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['Syne', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        threat:  { DEFAULT: '#ef4444', light: '#fca5a5', dim: 'rgba(239,68,68,0.12)' },
        warn:    { DEFAULT: '#f59e0b', light: '#fcd34d', dim: 'rgba(245,158,11,0.12)' },
        safe:    { DEFAULT: '#10b981', light: '#6ee7b7', dim: 'rgba(16,185,129,0.12)' },
        info:    { DEFAULT: '#3b82f6', light: '#93c5fd', dim: 'rgba(59,130,246,0.12)' },
        surface: { 50: '#1a1f2e', 100: '#151922', 200: '#0f1117', 300: '#090c12' },
        border:  { DEFAULT: 'rgba(255,255,255,0.07)', bright: 'rgba(255,255,255,0.14)' },
      },
      boxShadow: {
        'glow-red':   '0 0 20px rgba(239,68,68,0.25)',
        'glow-green': '0 0 20px rgba(16,185,129,0.25)',
        'glow-blue':  '0 0 20px rgba(59,130,246,0.25)',
        'card':       '0 4px 24px rgba(0,0,0,0.4)',
      },
    },
  },
  plugins: [],
};
