import React, { Component } from "react";
import "./App.css";
import "./css/main.css";
import "./css/util.css";
import { Register } from "./register.js";
import { Login } from "./login.js";
import { Home } from "./home.js";

export class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      signup: false,
      verify: false,
      home: true,
    };
    this.backhome = this.backhome.bind(this);
  }

  componentWillMount() {
    // define increment counter part
    const tabsOpen = localStorage.getItem("tabsOpen");
    const startTimestamp = localStorage.getItem("startTimestamp");
    const endTimeStamp = new Date().getTime();
    console.log("tabsOpen", tabsOpen);
    if (tabsOpen == null) {
      localStorage.setItem("tabsOpen", 1);
      localStorage.setItem("startTimestamp", endTimeStamp);
    } else {
      var timeDiff = endTimeStamp - startTimestamp;
      if (timeDiff < 30000) {
        localStorage.setItem("tabsOpen", parseInt(tabsOpen) + parseInt(1));
        if (tabsOpen > 10) {
          this.backhome();
        } else {
          this.setState({
            signup: false,
            verify: false,
            home: true,
          });
        }
      } else {
        this.setState({
          signup: false,
          verify: false,
          home: true,
        });
        localStorage.setItem("tabsOpen", 1);
        localStorage.setItem("startTimestamp", endTimeStamp);
      }
    }
  }

  backhome() {
    this.setState({
      signup: false,
      verify: false,
      home: false,
    });
  }
  signup() {
    this.setState({
      signup: true,
      verify: false,
      home: false,
    });
  }
  verify() {
    this.setState({
      signup: false,
      verify: true,
      home: false,
    });
  }
  render() {
    let home = (
      <div>
        <div className="limiter">
          <div className="container-login100">
            <span className="login100-form-title ">2FA to prevent DDoS</span>
            <div className="wrap-login100 p-l-110 p-r-110 p-t-62 p-b-33">
              <form className="login100-form validate-form flex-sb flex-w">
                <span className="login100-form-title p-b-53">
                  Login Or Register
                </span>
                <div className="container-login100-form-btn m-t-17">
                  <button
                    onClick={this.verify.bind(this)}
                    className="login100-form-btn"
                  >
                    Already registered? Login
                  </button>
                </div>
                <div className="container-login100-form-btn m-t-17">
                  <button
                    onClick={this.signup.bind(this)}
                    className="login100-form-btn"
                  >
                    New user? Register
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
        <div id="dropDownSelect1"></div>
      </div>
    );

    return (
      <div>
        {!this.state.signup && !this.state.verify && !this.state.home
          ? home
          : ""}
        {this.state.home ? <Home /> : ""}
        {this.state.signup ? <Register backhome={this.backhome} /> : ""}
        {this.state.verify ? <Login backhome={this.backhome} /> : ""}
      </div>
    );
  }
}
export default App;
