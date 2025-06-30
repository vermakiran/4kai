import React, { useEffect, useState } from 'react';
import MainLogo from '../assets/images/Main.png';
import { motion, AnimatePresence } from 'framer-motion';

const Splash = ({ onFinish }) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(onFinish, 500); // Wait for exit animation to complete
    }, 2000); // Show splash for 2 seconds

    return () => clearTimeout(timer);
  }, [onFinish]);

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: '#ffffff',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999
          }}
        >
          <motion.img
            src={MainLogo}
            alt="4kast.ai"
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ 
              duration: 0.8,
              ease: "easeOut"
            }}
            style={{
              maxWidth: '400px',
              width: '80%',
              height: 'auto'
            }}
          />
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.5 }}
            style={{
              marginTop: '20px',
              fontSize: '20px',
              color: '#002855',
              fontFamily: 'Inter, sans-serif',
              fontWeight: 500,
              textAlign: 'center',
              letterSpacing: '0.5px'
            }}
          >
            Your Intelligent Forecasting Solution
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default Splash; 