import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Bell, AlertTriangle, Info } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

function ReminderCard() {
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    checkNotifications();
    const interval = setInterval(checkNotifications, 60000); // Check every minute
    return () => clearInterval(interval);
  }, []);

  const checkNotifications = async () => {
    try {
      const res = await axios.get(`${API_BASE}/notifications/check?temperature=28`);
      setNotifications(res.data.notifications || []);

      // Show browser notification for critical alerts
      if (res.data.notifications?.length > 0) {
        const critical = res.data.notifications.find(n => n.level === 'critical');
        if (critical && Notification.permission === 'granted') {
          new Notification(critical.title, {
            body: critical.message,
            icon: '/logo192.png'
          });
        }
      }
    } catch (error) {
      console.error('Error checking notifications:', error);
    }
  };

  if (notifications.length === 0) {
    return (
      <div className="card reminder-card">
        <Bell size={20} />
        <span>No reminders - You're on track! ✅</span>
      </div>
    );
  }

  return (
    <div className="reminders-list">
      {notifications.map((notif, idx) => (
        <div key={idx} className={`card reminder-card level-${notif.level}`}>
          {notif.level === 'critical' && <AlertTriangle size={20} />}
          {notif.level === 'warning' && <Bell size={20} />}
          {notif.level === 'info' && <Info size={20} />}
          
          <div className="reminder-content">
            <strong>{notif.title}</strong>
            <p>{notif.message}</p>
            {notif.recommended_ml && (
              <div className="action-button">
                Drink {notif.recommended_ml}ml now
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export default ReminderCard;