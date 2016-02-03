import React from "react";
import Router from "./lib/router";

import FavIcon from "../images/favicon-96x96.png";
import FavIcon2 from "../images/favicon.ico";
import FavIcon3 from "../images/apple-touch-icon-114x114.png";

import Styles from "../styles/main.less";
import MaterializeCSS from "materialize-css";


Router.run(Handler => React.render(<Handler />, document.body));