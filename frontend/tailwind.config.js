/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        meteo: {
          900: '#070b14',
          850: '#0b1120',
          800: '#0f172a',
          750: '#15203b',
          700: '#1e293b',
          600: '#334155',
          500: '#64748b',
          400: '#94a3b8',
          100: '#f1f5f9',
          50: '#f8fafc',
        },
        cyclone: {
          blue: '#0284c7',
          cyan: '#06b6d4',
          teal: '#14b8a6',
          amber: '#f59e0b',
          rose: '#f43f5e',
          violet: '#8b5cf6',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}
