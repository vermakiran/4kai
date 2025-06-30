import React from 'react';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';

const PageTitle = ({ title }) => {
  return (
    <Box sx={{ 
      mb: 4,  // Increased bottom margin for better spacing
      mt: 2,  // Increased top margin for better spacing
      ml: 3,  // Increased left margin for better alignment
      borderBottom: '1px solid #e0e0e0', // Subtle bottom border
      pb: 2,  // Padding bottom for the border
    }}>
      <Typography 
        variant="h4" 
        sx={{ 
          color: '#002855',
          fontWeight: 600,
          fontSize: '24px',
          fontFamily: 'Inter, Roboto, Arial, sans-serif',
          letterSpacing: '-0.5px',
        }}
      >
        {title}
      </Typography>
    </Box>
  );
};

export default PageTitle; 