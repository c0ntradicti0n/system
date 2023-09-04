const path = require('path');


module.exports = {
  jest: {
    configure: {
      testEnvironment: 'jsdom',
      testMatch: [
        '<rootDir>/src/**/__tests__/**/*.{js,jsx,ts,tsx}',
        '<rootDir>/src/**/*.{spec,test}.{js,jsx,ts,tsx}',
        '<rootDir>/test/**/*.{js,jsx,ts,tsx}',
      ],
    },
  },
   webpack: {
        configure: (webpackConfig) => {
            // Add the custom loader rule to the existing Webpack config.
            webpackConfig.module.rules.push({
                test: path.resolve(__dirname, 'node_modules/leader-line/'),
                use: [{
                    loader: 'skeleton-loader',
                    options: { procedure: content => `${content}export default LeaderLine` }
                }]
            });

            return webpackConfig;
        }
    }
}
