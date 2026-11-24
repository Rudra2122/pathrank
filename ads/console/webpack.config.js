// Webpack 5: TypeScript via ts-loader, CSS extracted in production, dev server proxies the API to
// the Spring Boot ad server on :8090 so the browser never needs CORS.
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = (env, argv) => {
  const prod = argv.mode === 'production';
  return {
    entry: './src/index.tsx',
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: prod ? 'assets/[name].[contenthash:8].js' : 'assets/[name].js',
      clean: true,
      publicPath: '/',
    },
    resolve: { extensions: ['.tsx', '.ts', '.js'] },
    devtool: prod ? 'source-map' : 'eval-cheap-module-source-map',
    module: {
      rules: [
        { test: /\.tsx?$/, loader: 'ts-loader', exclude: /node_modules/, options: { onlyCompileBundledFiles: true } },
        { test: /\.css$/, use: [prod ? MiniCssExtractPlugin.loader : 'style-loader', 'css-loader'] },
      ],
    },
    plugins: [
      new HtmlWebpackPlugin({ template: './src/index.html' }),
      ...(prod ? [new MiniCssExtractPlugin({ filename: 'assets/[name].[contenthash:8].css' })] : []),
    ],
    performance: { hints: prod ? 'warning' : false },
    devServer: {
      port: Number(process.env.PORT || 5180),
      historyApiFallback: true,
      proxy: [{ context: ['/auth', '/api'], target: process.env.API || 'http://localhost:8090' }],
    },
  };
};
