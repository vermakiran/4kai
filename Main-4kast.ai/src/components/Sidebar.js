// import React from "react";
// import { NavLink } from "react-router-dom";
// import {
//   FaTachometerAlt,
//   FaComments,
//   FaUsers,
//   FaCogs,
//   FaDatabase,
//   FaChartBar,
//   FaProjectDiagram,
// } from "react-icons/fa";
// import "../App.css";

// const menuItems = [
//   { name: "Dashboard", path: "/", icon: <FaTachometerAlt /> },
//   { name: "Import Data", path: "/import", icon: <FaComments /> },
//   { name: "Forecast Settings", path: "/forecast-settings", icon: <FaUsers /> },
//   { name: "Planner Workbench", path: "/planner-workbench", icon: <FaUsers /> },
//   { name: "Outlier Report", path: "/outlier-report", icon: <FaCogs /> },
//   { name: "Master Data", path: "/master-data", icon: <FaDatabase /> },
//   { name: "I AM", path: "/iam", icon: <FaChartBar /> },
//   { name: "Integration Scenarios", path: "/integration-scenarios", icon: <FaProjectDiagram /> },
// ];

// function Sidebar() {
//   return (
//     <div className="sidebar">
//       <ul className="sidebar-menu">
//         {menuItems.map((item, index) => (
//           <li key={index} className="sidebar-item">
//             <NavLink
//               to={item.path}
//               className={({ isActive }) =>
//                 isActive ? "sidebar-link active" : "sidebar-link"
//               }
//             >
//               <span className="sidebar-icon">{item.icon}</span>
//               <span className="sidebar-text">{item.name}</span>
//             </NavLink>
//           </li>
//         ))}
//       </ul>
//     </div>
//   );
// }

// export default Sidebar;

// src/components/Sidebar.js

// src/components/Sidebar.js

import React from "react";
import { NavLink } from "react-router-dom";
import {
  FaTachometerAlt,
  FaComments,
  FaUsers,
  FaCogs,
  FaDatabase,
  FaChartBar,
  FaProjectDiagram,
  FaWrench,
} from "react-icons/fa";
import "../App.css";

const menuItems = [
  { name: "Dashboard", path: "/dashboard", icon: <FaTachometerAlt /> },
  { name: "Import Data", path: "/import", icon: <FaComments /> },
  { name: "Forecast Settings", path: "/forecast-settings", icon: <FaUsers /> },
  { name: "Planner Workbench", path: "/planner-workbench", icon: <FaWrench /> },
  // { name: "Outlier Report", path: "/outlier-report", icon: <FaCogs /> },
  // { name: "Integration Scenarios", path: "/integration-scenarios", icon: <FaProjectDiagram /> },
  { name: "Admin Settings", path: "/iam", icon: <FaChartBar /> },
];

function Sidebar() {
  return (
    <div className="sidebar">
      <ul className="sidebar-menu">
        {menuItems.map((item, index) => (
          <li key={index} className="sidebar-item">
            <NavLink
              to={item.path}
              className={({ isActive }) =>
                isActive ? "sidebar-link active" : "sidebar-link"
              }
            >
              <span className="sidebar-icon">{item.icon}</span>
              <span className="sidebar-text">{item.name}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Sidebar;