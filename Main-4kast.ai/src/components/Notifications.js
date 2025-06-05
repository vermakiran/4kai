import React from "react";
import { FaTimes } from "react-icons/fa";
import "../App.css";

function NotificationPopup({ togglePopup }) {
  // Sample notifications
  const newNotifications = [
    { id: 1, text: "📊 New Forecast Available - Forecast for Q2 is ready." },
    { id: 2, text: "📥 Data Import Successful - Latest sales data imported." },
    { id: 3, text: "🚨 Stock Alert - Inventory level for Product X is low!" },
  ];

  const oldNotifications = [
    { id: 4, text: "✅ Forecast Reviewed - Q1 forecast approved." },
    { id: 5, text: "🛠️ Settings Updated - Demand parameters modified." },
  ];

  return (
    <div className="notification-popup">
      <div className="popup-header">
        <h3>🔔 Notifications</h3>
        {/* This should now properly close the pop-up */}
        <FaTimes className="close-icon" onClick={togglePopup} />
      </div>

      {/* New Notifications */}
      <div className="notification-section">
        <h4>New Notifications</h4>
        {newNotifications.length > 0 ? (
          newNotifications.map((notification) => (
            <div key={notification.id} className="notification-item new">
              {notification.text}
            </div>
          ))
        ) : (
          <p className="no-notifications">No new notifications</p>
        )}
      </div>

      {/* Old Notifications */}
      <div className="notification-section">
        <h4>Old Notifications</h4>
        {oldNotifications.map((notification) => (
          <div key={notification.id} className="notification-item old">
            {notification.text}
          </div>
        ))}
      </div>
    </div>
  );
}

export default NotificationPopup;
